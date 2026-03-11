#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select static-structure-biased frames for COLMAP.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--target-count", type=int, default=72)
    parser.add_argument("--min-gap", type=int, default=1, help="Minimum frame index gap between selected frames.")
    return parser.parse_args()


def load_image(path: Path, max_width: int = 480) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.width > max_width:
            height = int(round(image.height * max_width / image.width))
            image = image.resize((max_width, height))
        rgb = np.asarray(image, dtype=np.float32) / 255.0
    gray = np.dot(rgb[..., :3], np.array([0.299, 0.587, 0.114], dtype=np.float32))
    return rgb, gray


def laplacian_variance(gray: np.ndarray) -> float:
    center = gray[1:-1, 1:-1]
    lap = (
        gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
        - 4.0 * center
    )
    return float(np.var(lap))


def edge_density(gray: np.ndarray) -> float:
    gy = np.abs(gray[1:, :] - gray[:-1, :])
    gx = np.abs(gray[:, 1:] - gray[:, :-1])
    return float((np.mean(gx > 0.12) + np.mean(gy > 0.12)) / 2.0)


def hi_vis_fraction(rgb: np.ndarray) -> float:
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    yellow = (r > 0.65) & (g > 0.6) & (b < 0.35)
    orange = (r > 0.7) & (g > 0.25) & (g < 0.65) & (b < 0.3)
    return float(np.mean(yellow | orange))


def zscore(values: list[float]) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std())
    if std < 1e-8:
        return [0.0 for _ in values]
    return [float((value - mean) / std) for value in arr]


def main() -> int:
    args = parse_args()
    paths = sorted(args.input_dir.glob("frame_*.jpg"))
    if not paths:
        raise SystemExit(f"No frames found in {args.input_dir}")

    metrics: list[dict[str, float | str | int]] = []
    previous_gray: np.ndarray | None = None

    for index, path in enumerate(paths):
        rgb, gray = load_image(path)
        blur = laplacian_variance(gray)
        edges = edge_density(gray)
        hi_vis = hi_vis_fraction(rgb)
        motion = 0.0 if previous_gray is None else float(np.mean(np.abs(gray - previous_gray)))
        previous_gray = gray
        metrics.append(
            {
                "path": str(path),
                "name": path.name,
                "index": index,
                "blur": blur,
                "edges": edges,
                "motion": motion,
                "hi_vis": hi_vis,
            }
        )

    blur_scores = zscore([float(item["blur"]) for item in metrics])
    edge_scores = zscore([float(item["edges"]) for item in metrics])
    motion_scores = zscore([float(item["motion"]) for item in metrics])
    hi_vis_scores = zscore([float(item["hi_vis"]) for item in metrics])

    for item, blur_score, edge_score, motion_score, hi_vis_score in zip(
        metrics, blur_scores, edge_scores, motion_scores, hi_vis_scores
    ):
        item["score"] = 1.5 * blur_score + 0.8 * edge_score - 1.3 * motion_score - 1.4 * hi_vis_score

    ranked = sorted(metrics, key=lambda item: float(item["score"]), reverse=True)
    selected: list[dict[str, float | str | int]] = []
    used_indices: list[int] = []

    for item in ranked:
        index = int(item["index"])
        if any(abs(index - used) <= args.min_gap for used in used_indices):
            continue
        selected.append(item)
        used_indices.append(index)
        if len(selected) >= args.target_count:
            break

    selected.sort(key=lambda item: int(item["index"]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for old_file in args.output_dir.glob("frame_*.jpg"):
        old_file.unlink()

    manifest: list[dict[str, object]] = []
    for new_index, item in enumerate(selected, start=1):
        source = Path(str(item["path"]))
        target = args.output_dir / f"frame_{new_index:05d}.jpg"
        shutil.copy2(source, target)
        manifest.append(
            {
                "source_name": item["name"],
                "target_name": target.name,
                "source_index": item["index"],
                "score": round(float(item["score"]), 4),
                "blur": round(float(item["blur"]), 4),
                "edges": round(float(item["edges"]), 4),
                "motion": round(float(item["motion"]), 4),
                "hi_vis": round(float(item["hi_vis"]), 6),
            }
        )

    report = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "input_count": len(paths),
        "selected_count": len(selected),
        "target_count": args.target_count,
        "min_gap": args.min_gap,
        "selected": manifest,
        "top_ranked_preview": [
            {
                "name": item["name"],
                "index": item["index"],
                "score": round(float(item["score"]), 4),
            }
            for item in ranked[:20]
        ],
    }
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2))
    print(f"Selected {len(selected)} frames into {args.output_dir}")
    print(f"Wrote report to {args.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
