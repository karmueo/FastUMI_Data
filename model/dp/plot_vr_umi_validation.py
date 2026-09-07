"""生成末端数据轨迹、固定批次过拟合和小规模验证曲线，保存独立 PNG 图。"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
import zarr


def plot_data(dataset_path, output):
    """绘制首段实际/目标末端轨迹和全部 episode 的步间变化分布。"""
    root = zarr.open_group(str(dataset_path), mode='r')
    data = root['data']
    ends = root['meta/episode_ends'][:]
    end = int(ends[0])
    times = data['timestamp'][:end]
    times -= times[0]
    observed = data['robot0_eef_pos'][:end]
    target = data['action'][:end]
    rotations = Rotation.from_rotvec(data['robot0_eef_rot_axis_angle'][:end])
    figure, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    for axis, label in enumerate(('x', 'y', 'z')):
        axes[0,0].plot(times, observed[:,axis], label=f'obs {label}')
        axes[0,0].plot(times, target[:,axis], '--', label=f'action {label}', alpha=.7)
    axes[0,0].set(title='Episode 0: base_link -> Link7 position', xlabel='Time (s)', ylabel='Position (m)')
    axes[0,0].legend(ncol=3)
    axes[0,1].plot(times, np.rad2deg((rotations[0].inv()*rotations).magnitude()), label='observed')
    axes[0,1].plot(times, np.rad2deg((rotations[0].inv()*Rotation.from_rotvec(target[:,3:6])).magnitude()), '--', label='target')
    axes[0,1].set(title='Orientation distance from initial pose', xlabel='Time (s)', ylabel='Angle (deg)')
    axes[0,1].legend()
    axes[1,0].plot(observed[:,0], observed[:,1], label='observed')
    axes[1,0].plot(target[:,0], target[:,1], '--', label='target')
    axes[1,0].set(title='XY path', xlabel='X (m)', ylabel='Y (m)')
    axes[1,0].axis('equal')
    axes[1,0].legend()
    axes[1,1].plot(times, data['robot0_gripper_width'][:end,0], label='state')
    axes[1,1].plot(times, target[:,-1], label='action')
    axes[1,1].set(title='Gripper (original normalized encoding)', xlabel='Time (s)', ylabel='Gripper value')
    axes[1,1].legend()
    position_steps, rotation_steps = [], []
    for start, end in zip(np.r_[0,ends[:-1]], ends):
        position_steps.extend(np.linalg.norm(np.diff(data['robot0_eef_pos'][start:end], axis=0), axis=-1))
        rotation = Rotation.from_rotvec(data['robot0_eef_rot_axis_angle'][start:end])
        rotation_steps.extend(np.rad2deg((rotation[:-1].inv()*rotation[1:]).magnitude()))
    axes[2,0].hist(np.asarray(position_steps)*1000, bins=80, log=True)
    axes[2,0].set(title='All episodes: adjacent position changes', xlabel='Distance (mm)', ylabel='Count (log)')
    axes[2,1].hist(rotation_steps, bins=80, log=True)
    axes[2,1].set(title='All episodes: adjacent rotation changes', xlabel='Angle (deg)', ylabel='Count (log)')
    figure.savefig(output/'data_trajectories.png', dpi=160)
    plt.close(figure)
    return {'position_step_m_quantiles': np.quantile(position_steps,[0,.5,.95,.99,1]).tolist(),
            'rotation_step_deg_quantiles': np.quantile(rotation_steps,[0,.5,.95,.99,1]).tolist()}


def plot_training(validation, output):
    """绘制验证指标，缺少尚未运行阶段时仅输出已经生成的曲线。"""
    overfit = validation/'overfit/metrics.jsonl'
    if overfit.exists():
        rows = [json.loads(line) for line in overfit.read_text().splitlines()]
        fig, axes = plt.subplots(1,2,figsize=(11,4), constrained_layout=True)
        for axis, key, title in ((axes[0],'val_loss','Fixed-batch diffusion loss'),
                                 (axes[1],'val_action_mse_error','Fixed-batch normalized action MSE')):
            axis.plot([row['step'] for row in rows], [row[key] for row in rows], marker='o')
            axis.set(title=title, xlabel='Optimizer step', yscale='log')
            axis.grid(alpha=.25)
        fig.savefig(output/'overfit_curves.png', dpi=160)
        plt.close(fig)
    small = validation/'small/logs.json.txt'
    if small.exists():
        rows = [json.loads(line) for line in small.read_text().splitlines()]
        rows = [row for row in rows if 'fixed_val_action_mse_error' in row]
        if rows:
            fig, axes = plt.subplots(1,2,figsize=(11,4), constrained_layout=True)
            epoch = [row['epoch']+1 for row in rows]
            axes[0].plot(epoch,[row['train_loss'] for row in rows],label='train')
            axes[0].plot(epoch,[row['val_loss'] for row in rows],label='validation (EMA)')
            axes[0].set(title='32 train / 8 validation episodes',xlabel='Epoch',ylabel='Diffusion loss',yscale='log')
            axes[0].legend()
            axes[1].plot(epoch,[row['fixed_val_action_mse_error'] for row in rows],marker='.')
            axes[1].set(title='Fixed validation samples',xlabel='Epoch',ylabel='Normalized action MSE',yscale='log')
            for axis in axes:
                axis.grid(alpha=.25)
            fig.savefig(output/'small_training_curves.png',dpi=160)
            plt.close(fig)


def main():
    """读取数据和验证目录，生成可分享的 PNG 图与数值统计。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True,type=Path)
    parser.add_argument('--validation',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    statistics = plot_data(args.dataset,args.output)
    plot_training(args.validation,args.output)
    (args.output/'trajectory_statistics.json').write_text(json.dumps(statistics,indent=2))
    print(json.dumps(statistics,indent=2))


if __name__ == '__main__':
    main()
