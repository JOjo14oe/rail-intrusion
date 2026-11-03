"""One entry point for the literature pipeline in out/lit/ (search -> pdf -> text).

Why this exists: the stage scripts in out/lit/ are one-shot experiments with
hardcoded inputs, so "run the paper thing" used to mean remembering six file
names and their order, in a shell that must have the dead 127.0.0.1:7897 proxy
stripped.  This script is only an orchestrator: it runs the existing stages
in order, skips a stage whose artifact is already newer than the stage script,
tees console output to out/lit/logs/, and prints what is still missing.

ponytail: the stage scripts themselves stay in out/lit/ (unversioned) because
they carry absolute paths and their artifacts (results/locations/abstracts/
authors .json) sit next to them; moving them would rewrite 3 path consts for
no functional gain.  Upgrade path: when a stage next needs editing, move it to
tools/lit/ and make its output dir an argv.

Usage:
  python tools/run_lit.py                 # all stages, skip up-to-date ones
  python tools/run_lit.py --force         # rerun every stage
  python tools/run_lit.py --steps search download
  python tools/run_lit.py --list
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The stage scripts and their json artifacts live one level UP, in the workspace
# scratch dir (Deepseek Harness/out/lit), not inside the repo.
LIT = ROOT.parent / "out" / "lit"
REFS = ROOT / "refs"
TEXT = REFS / "extracted-text"
LOGS = LIT / "logs"

# (name, script, artifact that marks the stage as done)
STAGES = [
    ("search", "lit_search.py", "results.json"),
    ("resolve", "resolve.py", "locations.json"),
    ("download", "download.py", None),  # done-ness lives in refs/*.pdf
    ("abstracts", "abstracts.py", "abstracts.json"),
    ("authors", "authors.py", "authors.json"),
    ("extract", "extract.py", None),  # done-ness lives in refs/extracted-text/
]


def clean_env() -> dict[str, str]:
    env = dict(os.environ)
    for key in list(env):
        if key.upper() in {"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"}:
            env.pop(key, None)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def is_fresh(stage: str, artifact: str | None) -> bool:
    script = LIT / dict((s[0], s[1]) for s in STAGES)[stage]
    if artifact is None:
        script_mtime = script.stat().st_mtime if script.exists() else 0
        if stage == "download":
            pdfs = list(REFS.glob("*.pdf"))
            return bool(pdfs) and min(p.stat().st_mtime for p in pdfs) > script_mtime
        if stage == "extract":
            pdfs = list(REFS.glob("*.pdf"))
            txts = list(TEXT.glob("*.txt"))
            return len(txts) >= len(pdfs) and bool(txts) and min(t.stat().st_mtime for t in txts) > script_mtime
        return False
    target = LIT / artifact
    if not target.exists() or not script.exists():
        return False
    return target.stat().st_mtime > script.stat().st_mtime


def run_stage(name: str, script_name: str) -> int:
    script = LIT / script_name
    if not script.exists():
        print(f"[lit] MISSING stage script: {script}")
        return 1
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"{name}-{time.strftime('%Y%m%d-%H%M%S')}.log"
    print(f"[lit] {name}: {script_name} -> {log.name}")
    t0 = time.time()
    with log.open("w", encoding="utf-8", errors="replace") as fh:
        proc = subprocess.Popen(
            [sys.executable, "-u", str(script)],
            cwd=str(LIT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=clean_env(),
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        tail: list[str] = []
        for line in proc.stdout:
            fh.write(line)
            tail.append(line.rstrip())
            if len(tail) > 400:
                tail.pop(0)
        code = proc.wait()
    for line in tail[-8:]:
        print(f"    | {line}")
    print(f"[lit] {name}: exit={code} {time.time() - t0:.0f}s")
    return code


def summary() -> None:
    def count_json(name: str, key: str | None) -> str:
        path = LIT / name
        if not path.exists():
            return "missing"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            return f"unreadable ({exc})"
        if isinstance(data, dict) and key:
            inner = data.get(key)
            if isinstance(inner, dict):
                return f"{len(inner)} records"
        if isinstance(data, dict):
            return f"{len(data)} keys"
        if isinstance(data, list):
            return f"{len(data)} items"
        return "?"

    pdfs = sorted(REFS.glob("*.pdf"))
    txts = sorted(TEXT.glob("*.txt"))
    size_mb = sum(p.stat().st_size for p in pdfs) / 1e6
    print("\n[lit] === artifacts ===")
    print(f"    search    results.json   {count_json('results.json', 'results')}")
    print(f"    resolve   locations.json {count_json('locations.json', None)}")
    print(f"    abstracts abstracts.json {count_json('abstracts.json', None)}")
    print(f"    authors   authors.json   {count_json('authors.json', None)}")
    print(f"    pdfs      refs/*.pdf     {len(pdfs)} files, {size_mb:.1f} MB")
    print(f"    text      extracted-text {len(txts)} files")
    missing = [p.name for p in pdfs if not (TEXT / (p.stem + ".txt")).exists()]
    if missing:
        print(f"    NOT EXTRACTED ({len(missing)}): {', '.join(missing[:5])}...")
    if len(txts) < len(pdfs):
        print("    -> run again without --force to finish extraction")


def main() -> int:
    names = [s[0] for s in STAGES]
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", nargs="+", choices=names, default=names)
    ap.add_argument("--force", action="store_true", help="rerun stages even when their artifact is fresh")
    ap.add_argument("--list", action="store_true", help="print the stage order and exit")
    args = ap.parse_args()

    if args.list:
        for name, script, artifact in STAGES:
            print(f"{name:10s} {script:16s} -> {artifact or 'refs/ + extracted-text/'}")
        return 0

    if not LIT.exists():
        print(f"[lit] stage dir missing: {LIT}")
        return 1

    failures = []
    for name, script, artifact in STAGES:
        if name not in args.steps:
            continue
        if not args.force and is_fresh(name, artifact):
            print(f"[lit] {name}: up to date, skipped (use --force)")
            continue
        if run_stage(name, script) != 0:
            failures.append(name)

    summary()
    if failures:
        print(f"\n[lit] failed stages: {failures}")
        return 1
    print("\n[lit] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
