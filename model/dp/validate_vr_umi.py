"""执行 Link7 UMI 固定批次过拟合与 32/8 段收敛验证；不启动全量训练。"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import hydra
from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import default_collate
import zarr

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.sampler import get_val_mask
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.workspace.train_diffusion_unet_image_workspace import TrainDiffusionUnetImageWorkspace, evaluate_policy


def compose_config(dataset, split):
    """组合原 UMI 派生配置，显式指定数据路径和无重叠验证划分。"""
    with initialize_config_dir(version_base=None, config_dir=str(Path(__file__).resolve().parent/'diffusion_policy/config')):
        cfg = compose(config_name='train_diffusion_unet_timm_vr_umi_workspace')
    cfg.task.dataset_path = str(dataset.resolve())
    cfg.task.dataset.train_episode_indices = split['train_episode_indices']
    cfg.task.dataset.val_episode_indices = split['val_episode_indices']
    OmegaConf.resolve(cfg)
    return cfg


def select_splits(dataset):
    """从固定 180/20 主划分内选择 32/8 子集，返回索引和原始 episode 名。"""
    root = zarr.open_group(str(dataset), mode='r')
    names = root['meta/episode_names'][:].tolist()
    if len(names) != 200:
        raise ValueError('Convergence experiment expects 200 episodes')
    mask = get_val_mask(200, 0.1, 42)
    rng = np.random.default_rng(42)
    train = sorted(rng.choice(np.flatnonzero(~mask), 32, replace=False).tolist())
    val = sorted(rng.choice(np.flatnonzero(mask), 8, replace=False).tolist())
    return {'train_episode_indices': train, 'val_episode_indices': val,
            'train_episodes': [names[index] for index in train], 'val_episodes': [names[index] for index in val],
            'full_train_episode_indices': np.flatnonzero(~mask).tolist(),
            'full_val_episode_indices': np.flatnonzero(mask).tolist()}


def fixed_batch_overfit(dataset_path, output, split):
    """冻结视觉编码器，对固定四样本优化 1000 次并比较固定种子动作 MSE。"""
    output.mkdir()
    cfg = compose_config(dataset_path, split)
    cfg.policy.obs_encoder.transforms = None
    cfg.training.freeze_encoder = True
    cfg.training.num_epochs = 1
    workspace = TrainDiffusionUnetImageWorkspace(cfg, output_dir=str(output))
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    normalizer = dataset.get_normalizer()
    indices = np.linspace(0, len(dataset)-1, 4, dtype=int)
    batch = default_collate([dataset[int(index)] for index in indices])
    batch = dict_apply(batch, lambda value: value.to('cuda:0'))
    workspace.model.set_normalizer(normalizer)
    workspace.ema_model.set_normalizer(normalizer)
    workspace.model.to('cuda:0')
    workspace.ema_model.to('cuda:0')
    workspace.model.obs_encoder.requires_grad_(False)
    ema = hydra.utils.instantiate(cfg.ema, model=workspace.ema_model)
    scheduler = get_scheduler(workspace.cfg.training.lr_scheduler, workspace.optimizer,
                              num_warmup_steps=10, num_training_steps=1000)
    workspace.ema_model.eval()
    initial = evaluate_policy(workspace.ema_model, [batch], 'cuda:0', sample=True)
    history = [{'step': 0, **initial}]
    with (output/'metrics.jsonl').open('w', buffering=1) as log:
        log.write(json.dumps(history[0])+'\n')
        for step in range(1, 1001):
            workspace.model.train()
            workspace.model.obs_encoder.eval()
            workspace.optimizer.zero_grad(set_to_none=True)
            loss = workspace.model(batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite overfit loss at step {step}')
            loss.backward()
            workspace.optimizer.step()
            scheduler.step()
            ema.step(workspace.model)
            if step % 100 == 0:
                workspace.ema_model.eval()
                metrics = evaluate_policy(workspace.ema_model, [batch], 'cuda:0', sample=True)
                record = {'step': step, 'train_loss': float(loss.item()), **metrics}
                history.append(record)
                log.write(json.dumps(record)+'\n')
                print('OVERFIT', record, flush=True)
    workspace.global_step = 1000
    workspace.epoch = 1
    workspace.save_checkpoint(tag='latest', use_thread=False)
    reduction = 1 - history[-1]['val_action_mse_error'] / initial['val_action_mse_error']
    report = {'steps': 1000, 'sample_indices': indices.tolist(), 'initial': initial,
              'final': history[-1], 'action_mse_reduction': reduction, 'passed': reduction >= 0.8}
    (output/'result.json').write_text(json.dumps(report, indent=2))
    OmegaConf.save(cfg, output/'config.yaml')
    return report


def convergence_report(output):
    """按首末各三轮均值判断验证 loss 和固定验证样本 MSE 是否均下降 30%。"""
    records = [json.loads(line) for line in (output/'logs.json.txt').read_text().splitlines()]
    epochs = [row for row in records if 'fixed_val_action_mse_error' in row]
    reductions = {}
    for key in ('val_loss', 'fixed_val_action_mse_error'):
        initial = float(np.mean([row[key] for row in epochs[:3]]))
        final = float(np.mean([row[key] for row in epochs[-3:]]))
        reductions[key] = {'initial_three_mean': initial, 'final_three_mean': final,
                           'reduction': 1 - final/initial}
    report = {'epochs': len(epochs), 'metrics': reductions,
              'passed': len(epochs) >= 20 and all(value['reduction'] >= 0.3 for value in reductions.values())}
    return report


def main():
    """运行所选验证阶段；输出报告并以非零退出码表示验收未通过。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--stage', choices=('overfit','small'), required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    split = select_splits(args.dataset)
    split_path = args.output/'split.json'
    if split_path.exists() and json.loads(split_path.read_text()) != split:
        raise ValueError('Existing split differs from reproducible selection')
    split_path.write_text(json.dumps(split, indent=2))
    if args.stage == 'overfit':
        report = fixed_batch_overfit(args.dataset, args.output/'overfit', split)
    else:
        if not json.loads((args.output/'overfit/result.json').read_text())['passed']:
            raise ValueError('Fixed-batch overfit must pass before small training')
        output = (args.output/'small').resolve()
        output.mkdir()
        cfg = compose_config(args.dataset, split)
        cfg.training.num_epochs = 40
        # 先检查第 20 轮；未达到约定标准时继续，最多 40 轮。
        OmegaConf.update(cfg, 'training.convergence_stop_after', 20, force_add=True)
        cfg.training.sample_every = 5
        cfg.training.lr_warmup_steps = 2000
        config_path = args.output/'small_config.yaml'
        OmegaConf.save(cfg, config_path)
        command = [sys.executable, 'train.py', '--config-dir', str(args.output.resolve()),
                   '--config-name', 'small_config', f'hydra.run.dir={output}']
        environment = dict(os.environ)
        environment.pop('PYTHONPATH', None)
        environment.update(WANDB_MODE='offline', WANDB_DIR=str(output), HF_HUB_OFFLINE='1',
                           OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        (args.output/'small_command.json').write_text(json.dumps(command, indent=2))
        with (output/'console.log').open('w') as log:
            subprocess.run(command, cwd=Path(__file__).resolve().parent, env=environment,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        report = convergence_report(output)
        (output/'result.json').write_text(json.dumps(report, indent=2))
        # 无论学习效果是否通过，都保存最佳 checkpoint 的独立加载评估结果。
        command = [sys.executable, 'evaluate_vr_umi.py', '--checkpoint', str(output/'checkpoints/best.ckpt'),
                   '--dataset', str(args.dataset.resolve()), '--output', str(output/'evaluation')]
        with (output/'evaluation.log').open('w') as log:
            subprocess.run(command, cwd=Path(__file__).resolve().parent, env=environment,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    print(json.dumps(report, indent=2), flush=True)
    if not report['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
