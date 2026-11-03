"""序列发现：让数据集构建脚本能吃掉任意多个已解压的 OSDaR23 子序列。

为什么要测：OSDaR23 每个子序列单独一个 zip、单独一个 `<名称>_labels.json`，
且 1_calibration 的标注放在 `seq_1.2/` 子目录里而图像在 `1_calibration_1.2/`。
加序列时如果硬编码，每加一个 zip 就要改一次代码——这里把它变成纯函数并用临时目录验证。
"""

from __future__ import annotations

import json
from pathlib import Path

from railintrude.osdar23 import discover_sequences


def _touch_json(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"openlabel": {}}), encoding="utf-8")


def test_discovers_sequence_with_labels_in_sibling_dir(tmp_path: Path) -> None:
    # 1_calibration 的形态：标注在 seq_1.2/ 下，图像在 1_calibration_1.2/ 下
    _touch_json(tmp_path / "seq_1.2" / "1_calibration_1.2_labels.json")
    (tmp_path / "1_calibration_1.2" / "rgb_center").mkdir(parents=True)

    found = discover_sequences(tmp_path)

    assert [name for name, _img, _label in found] == ["1_calibration_1.2"]
    name, img_dir, label_json = found[0]
    assert img_dir == tmp_path / "1_calibration_1.2"
    assert label_json.name == "1_calibration_1.2_labels.json"


def test_discovers_sequence_with_labels_inside_its_own_dir(tmp_path: Path) -> None:
    _touch_json(tmp_path / "7_approach_underground_station_7.1" / "7_approach_underground_station_7.1_labels.json")
    _touch_json(tmp_path / "2_station_berliner_tor_2.1" / "2_station_berliner_tor_2.1_labels.json")

    found = discover_sequences(tmp_path)

    assert [name for name, _img, _label in found] == [
        "2_station_berliner_tor_2.1",
        "7_approach_underground_station_7.1",
    ]
    assert all(img_dir.name == name for name, img_dir, _label in found)


def test_reports_sequence_whose_images_are_not_extracted(tmp_path: Path) -> None:
    # 只解了标注、还没解图像：要照实返回（调用方据此打印 skip），不能静默消失。
    # 用 seq_x/ 这种“标注不在自己目录里”的形态，才是真的没有图像目录。
    _touch_json(tmp_path / "seq_x" / "9_station_ruebenkamp_9.1_labels.json")

    found = discover_sequences(tmp_path)

    assert len(found) == 1
    name, img_dir, _label = found[0]
    assert name == "9_station_ruebenkamp_9.1"
    assert not img_dir.is_dir()


def test_duplicate_label_copies_yield_one_entry_preferring_own_dir(tmp_path: Path) -> None:
    # 实测坑：同一个标注可能既在 seq_1.2/ 又在 1_calibration_1.2/ 里（解压工具复制过），
    # 不去重就会把同一序列处理两遍、图像重名互相覆盖。
    _touch_json(tmp_path / "seq_1.2" / "1_calibration_1.2_labels.json")
    _touch_json(tmp_path / "1_calibration_1.2" / "1_calibration_1.2_labels.json")
    (tmp_path / "1_calibration_1.2" / "rgb_center").mkdir(parents=True)

    found = discover_sequences(tmp_path)

    assert [name for name, _img, _label in found] == ["1_calibration_1.2"]
    _name, _img_dir, label_json = found[0]
    assert label_json.parent.name == "1_calibration_1.2"


def test_empty_root_returns_empty_list(tmp_path: Path) -> None:
    assert discover_sequences(tmp_path) == []
