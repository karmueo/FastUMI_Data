"""验证夹爪预览的对称开合以及正式七轴模型兼容性。"""
from pathlib import Path
import runpy
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PREVIEW = runpy.run_path(str(ROOT / 'datatool/view_rm75.launch.py'))


def test_preview_joints_and_arm_compatibility():
    """正式模型保持七轴，预览增加一个自由夹爪关节及一个跟随关节。"""
    robot = ET.parse(ROOT / 'assets/rm_75.urdf').getroot()
    arm_before = [ET.tostring(j) for j in robot.findall('joint') if j.get('type') != 'fixed']
    assert len(arm_before) == 7
    PREVIEW['enable_gripper_preview'](robot)
    assert arm_before == [ET.tostring(j) for j in robot.findall('joint') if j.get('type') == 'revolute']
    moving = robot.findall("joint[@type='prismatic']")
    assert len(moving) == 2
    assert moving[1].find('mimic').get('joint') == moving[0].get('name')
    assert moving[1].find('mimic').get('multiplier') == '1'


def test_finger_motion_and_mesh_clearance():
    """验证开到上限还原 CAD、左右对称移动，闭合后接触垫不越过中心面。"""
    robot = ET.parse(ROOT / 'assets/rm_75.urdf').getroot()
    PREVIEW['enable_gripper_preview'](robot)
    travel = PREVIEW['FINGER_TRAVEL_M']
    for side, direction in [('left', 1), ('right', -1)]:
        joint = robot.find(f"joint[@name='gripper_{side}_finger_joint']")
        origin = np.fromstring(joint.find('origin').get('xyz'), sep=' ')
        axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
        np.testing.assert_allclose(origin + axis * travel, 0, atol=1e-12)
        assert float(joint.find('limit').get('lower')) == 0
        assert float(joint.find('limit').get('upper')) == travel
        path = ROOT / f'assets/meshes/rm75_gripper_{side}.stl'
        data = path.read_bytes()
        count = int.from_bytes(data[80:84], 'little')
        assert len(data) == 84 + 50 * count
        vertices = np.frombuffer(data, dtype=np.dtype([
            ('normal', '<f4', 3), ('vertices', '<f4', (3, 3)), ('attribute', '<u2'),
        ]), offset=84)['vertices'].reshape(-1, 3) * .001
        assert np.isfinite(vertices).all()
        # 指端接触区域在原始 STEP Z>90 mm，排除底部交错滑块。
        tips = vertices[vertices[:, 2] > .09]
        assert len(tips) > 0
        for openness in (0, .5, 1):
            moved = tips + origin + axis * travel * openness
            assert np.min(direction * moved[:, 0]) >= 0
