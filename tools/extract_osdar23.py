"""从 OSDaR23 序列 zip 里按需取出需要的流，避免解压 8.8 GB。

用法：
  python tools/extract_osdar23.py "C:/.../7_approach_underground_station_7.1.zip" --streams ir_center rgb_center

只解压：标注 json、calibration.txt、readme.md、license.md，以及 --streams 指定的相机目录。
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

META_SUFFIXES = (".json", "calibration.txt", "readme.md", "license.md")


def is_meta(name: str) -> bool:
    base = name.rsplit("/", 1)[-1]
    return base in ("calibration.txt", "readme.md", "license.md") or base.endswith(".json")


def in_stream(name: str, streams) -> bool:
    """zip 内部有两种布局：<序列>/<流>/xxx.png 与 <流>/xxx.png。

    早期实现只认前者（startswith），导致 17 个扁平布局的序列解出 0 个文件却报“命中 14 个”。
    """
    want = set(streams)
    return any(part in want for part in name.split("/")[:-1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("zip_path")
    ap.add_argument("--dest", default=None, help="默认 data/osdar23/<zip 主名>")
    ap.add_argument("--streams", nargs="*", default=["ir_center", "rgb_center"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    zp = Path(args.zip_path)
    if not zp.is_file():
        print(f"找不到 {zp}")
        return 2
    with zp.open("rb") as fh:
        magic = fh.read(2)
    if magic != b"PK":
        print(f"不是 zip（前 2 字节 = {magic!r}）")
        return 2

    dest = Path(args.dest) if args.dest else Path("data/osdar23") / zp.stem
    print(f"zip: {zp}  ({zp.stat().st_size / 2**30:.2f} GiB)  ->  {dest}")

    want = set(args.streams)

    def in_stream(name: str) -> bool:
        # 两种内部布局都要支持：<序列>/<流>/xxx.png 与 <流>/xxx.png
        parts = name.split("/")[:-1]
        return any(p in want for p in parts)

    picked = []
    with zipfile.ZipFile(zp) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            name = info.filename
            if is_meta(name) or in_stream(name, args.streams):
                picked.append(info)
        total = sum(i.file_size for i in picked)
        print(f"命中 {len(picked)} 个文件，解压后约 {total / 2**20:.0f} MiB")
        if args.dry_run:
            for i in picked[:10]:
                print(f"  {i.filename}  {i.file_size}")
            return 0
        dest.mkdir(parents=True, exist_ok=True)
        for i in picked:
            target = dest / i.filename
            if target.is_file() and target.stat().st_size == i.file_size:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(i) as src, target.open("wb") as out:
                while chunk := src.read(1 << 20):
                    out.write(chunk)
    print("完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
