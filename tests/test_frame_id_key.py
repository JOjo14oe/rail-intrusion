"""回归：frame id 与 pcd 文件名的零填充不一致（labels "65" vs 文件 "065_...pcd"）"""
from railintrude.kitti import frame_id_key


def test_padded_pcd_name_and_plain_label_id_agree():
    assert frame_id_key("065_1631176102.632767000.pcd") == 65
    assert frame_id_key("65") == 65


def test_non_numeric_returns_none():
    assert frame_id_key("abc_1.pcd") is None
