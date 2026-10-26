"""部署指纹与标定测试，只读取小配置，不加载真实模型。"""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from dp_infer_lerobot.config import DiffusionConfig, SUPPORTED, validate_steps
from dp_infer_lerobot.deployment import (correction_from_profile, load_correction, load_profile,
                                         matrix_sha256, validate_urdf)

CONFIG = Path(__file__).resolve().parents[1] / 'config'


def test_profile_matches_new_training_calibration_and_installed_urdf():
    profile = load_profile(CONFIG / 'jingbao_deployment.yaml')
    cal = profile['calibration']
    assert cal['T_Cd_Link7_training_sha256'] == cal['T_Cd_Link7_deploy_sha256']
    # Existing correction generator uses standard json separators, not compact JSON.
    expected = hashlib.sha256(json.dumps(np.round(cal['T_Cd_Link7_training'], 12).tolist()).encode()).hexdigest()
    assert matrix_sha256(cal['T_Cd_Link7_training']) == expected
    correction = load_correction(CONFIG / 'link7_correction.yaml', profile)
    assert np.degrees(Rotation.from_matrix(correction[:3, :3]).magnitude()) == pytest.approx(1.015371, abs=1e-5)
    np.testing.assert_allclose(correction[:3, 3] * 1000, [-1.374356, 1.813137, -2.402907], atol=1e-5)
    validate_urdf(CONFIG.parents[1] / 'fastumi_rm75/assets/rm_75_kinematic.urdf', profile)


def test_reject_old_model_correction_and_modified_measurement():
    profile = load_profile(CONFIG / 'jingbao_deployment.yaml')
    with pytest.raises(ValueError, match='fingerprint'):
        load_correction(CONFIG.parents[1] / 'dp_infer/config/link7_correction.yaml', profile)
    bad = copy.deepcopy(profile)
    bad['calibration']['T_Cd_Link7_training'][0][3] += .01
    with pytest.raises(ValueError, match='fingerprint'):
        correction_from_profile(bad)


def test_only_supported_model_configuration_is_accepted(tmp_path):
    cfg = copy.deepcopy(SUPPORTED)
    cfg['num_inference_steps'] = 16
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(cfg))
    assert DiffusionConfig.from_file(path).action_feature.shape == (10,)
    cfg['input_features']['observation.state']['shape'] = [7]
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match='input_features'):
        DiffusionConfig.from_file(path)
    for steps in (0, 51, True, 8.5):
        with pytest.raises(ValueError):
            validate_steps(steps)


def test_checkpoint_binding_rejects_mixed_weights_and_external_processor(tmp_path):
    from dp_infer_lerobot.deployment import REQUIRED_FILES, sha256_file, validate_checkpoint
    from dp_infer_lerobot.processors import FEATURES, NORM_MAP

    pre_name, post_name = 'pre.safetensors', 'post.safetensors'
    normalize = {'registry_name': 'normalizer_processor', 'state_file': pre_name,
                 'config': {'eps': 1e-8, 'features': FEATURES, 'norm_map': NORM_MAP}}
    device = {'registry_name': 'device_processor', 'config': {'device': 'cuda', 'float_dtype': None}}
    pre = {'name': 'policy_preprocessor', 'steps': [
        {'registry_name': 'rename_observations_processor', 'config': {'rename_map': {}}},
        {'registry_name': 'to_batch_processor', 'config': {}}, device, normalize]}
    post = {'name': 'policy_postprocessor', 'steps': [
        {'registry_name': 'unnormalizer_processor', 'state_file': post_name,
         'config': {'eps': 1e-8, 'features': {'action': FEATURES['action']}, 'norm_map': NORM_MAP}}, device]}
    for name in REQUIRED_FILES | {pre_name, post_name}:
        (tmp_path / name).write_bytes(b'synthetic-file-for-fingerprint-test')
    (tmp_path / 'normalization_stats.json').write_text(json.dumps({'source': 'train'}))
    (tmp_path / 'policy_preprocessor.json').write_text(json.dumps(pre))
    (tmp_path / 'policy_postprocessor.json').write_text(json.dumps(post))
    profile = {'checkpoint_files': {p.name: sha256_file(p) for p in tmp_path.iterdir()}}
    validate_checkpoint(tmp_path, profile)
    (tmp_path / 'model.safetensors').write_bytes(b'other-model')
    with pytest.raises(ValueError, match='fingerprint'):
        validate_checkpoint(tmp_path, profile)
    pre['steps'][-1]['state_file'] = '../outside.safetensors'
    (tmp_path / 'policy_preprocessor.json').write_text(json.dumps(pre))
    with pytest.raises(ValueError, match='local'):
        validate_checkpoint(tmp_path, profile)
