"""只测试离线报告比较逻辑，不导入 Torch、不运行模型。"""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('dp_comparison', Path(__file__).resolve().parents[1] / 'tools/compare_policy.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_comparison_reports_mismatch_and_rejects_different_input(tmp_path):
    keys = ('normalized_image', 'normalized_state', 'features', 'conditioning',
            'denoiser_first', 'actions_normalized', 'actions')
    data = {key: np.zeros((1, 2), dtype=np.float32) for key in (*keys, 'image', 'state', 'noise')}
    meta = {'mode': 'export-reference', 'steps': 16, 'checkpoint_files': {'model': 'test'}}
    ref, candidate = tmp_path / 'reference.npz', tmp_path / 'candidate.npz'
    np.savez(ref, metadata=np.array(json.dumps(meta)), **data)
    meta['mode'] = 'export-candidate'
    np.savez(candidate, metadata=np.array(json.dumps(meta)), **data)
    args = SimpleNamespace(reference=ref, candidate=candidate, report=tmp_path / 'report.json', atol=1e-5, rtol=1e-4)
    assert module.compare(args) == 0
    data['actions'][0, 0] = .01
    np.savez(candidate, metadata=np.array(json.dumps(meta)), **data)
    assert module.compare(args) == 1
    assert not json.loads(args.report.read_text())['stages']['actions']['passed']
    data['noise'][0, 0] = 1
    np.savez(candidate, metadata=np.array(json.dumps(meta)), **data)
    with pytest.raises(ValueError, match='Inputs differ'):
        module.compare(args)
