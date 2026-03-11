#!/usr/bin/env python3

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse

from recycling_mrf.plan_viz import export_visualization_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export Mermaid and summary artifacts from an episode plan JSON.")
    parser.add_argument("--plan", type=Path, required=True, help="Path to episode_plan.json.")
    parser.add_argument("--output-dir", type=Path, help="Directory for visualization artifacts. Defaults to <plan>/viz.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    files = export_visualization_bundle(args.plan, args.output_dir)
    for name, path in files.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
