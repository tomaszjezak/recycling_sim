from __future__ import annotations

import argparse
from pathlib import Path

from recycling_mrf.config import SimulationConfig
from recycling_mrf.isaac_runner import IsaacConveyorRunner, IsaacSimUnavailableError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the MRF conveyor simulation.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the simulation JSON config.")
    parser.add_argument("--dry-run", action="store_true", help="Generate deterministic metadata without launching Isaac Sim.")
    parser.add_argument("--headless", action="store_true", help="Run Isaac Sim without a visible window.")
    parser.add_argument(
        "--viewport",
        choices=("overview", "camera", "overhead"),
        default="overview",
        help="When running with a window, show the overview camera, the configured camera, or an overhead conveyor camera.",
    )
    parser.add_argument(
        "--bare-bones",
        action="store_true",
        help="Use the lightest interactive scene setup: no sensor camera and no decorative item visuals.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Continuously replay short clean cycles without rebuilding the scene.",
    )
    parser.add_argument(
        "--loop-cycle-seconds",
        type=float,
        default=12.0,
        help="Duration of one looping cycle before spawned items are cleared and replayed.",
    )
    parser.add_argument(
        "--detections",
        action="store_true",
        help="Show simulated detection bounding boxes for selected conveyor items.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.loop_cycle_seconds <= 0:
        print("--loop-cycle-seconds must be positive")
        return 2
    config = SimulationConfig.from_file(args.config)
    runner = IsaacConveyorRunner(
        config=config,
        headless=args.headless,
        viewport_mode=args.viewport,
        bare_bones=args.bare_bones,
        loop=args.loop,
        loop_cycle_seconds=args.loop_cycle_seconds,
        detections=args.detections,
    )

    try:
        plan = runner.dry_run() if args.dry_run else runner.run()
    except IsaacSimUnavailableError as exc:
        print(exc)
        return 2

    print(
        f"Generated {len(plan.events)} spawn events in {config.output_dir / 'episode_plan.json'}"
    )
    return 0
