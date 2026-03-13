# Recycling MRF Conveyor Prototype

This repository contains a config-driven mixed recycling facility prototype for a single-stream
MRF line. It is built around `NVIDIA Isaac Sim`, but the non-Omniverse parts are usable without
Isaac installed so facility planning, config validation, and deterministic tests can run locally.

## What is implemented

- Config-driven facility inputs for one main conveyor and ordered station zones
- Deterministic mixed-item spawn planning plus per-item routing/disposition metadata
- A lazy Isaac Sim runner that overlays one recycling line into Isaac Sim's `Simple Warehouse`
- A dry-run mode that exports full facility/item lifecycle metadata without launching Isaac Sim
- Richer recycler-specific item visuals layered on top of simple physics proxies
- Unit tests for config validation, routing behavior, and deterministic planning

## Repo layout

- `configs/conveyor_demo.json`: default simulation config
- `configs/edco_conveyor_segment_a.json`: narrow conveyor mini-scene based on the solved `belt_a` segment
- `scripts/run_conveyor.py`: CLI entrypoint
- `src/recycling_mrf/config.py`: config model and validation
- `src/recycling_mrf/scenario.py`: deterministic facility episode planning and routing
- `src/recycling_mrf/scene.py`: facility scene metadata generation
- `src/recycling_mrf/isaac_runner.py`: Isaac Sim facility scene assembly and simulation loop
- `tests/`: config and scenario tests

## Local usage

Dry-run without Isaac Sim:

```bash
PYTHONPATH=src python3 scripts/run_conveyor.py --config configs/conveyor_demo.json --dry-run
```

Export Mermaid/text visualization artifacts from the dry-run output:

```bash
PYTHONPATH=src python3 scripts/export_plan_viz.py --plan outputs/conveyor_demo/episode_plan.json
```

Run unit tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

## Isaac Sim usage

Install Isaac Sim separately, then launch the script with Isaac Sim's Python environment
or a Python where Isaac packages are available:

```bash
PYTHONPATH=src python3 scripts/run_conveyor.py --config configs/conveyor_demo.json
```

For this repo on this machine, Isaac Sim is installed into `.venv-isaacsim`. The default
config loads Isaac Sim's built-in `Simple Warehouse` environment and places the conveyor,
stations, and capture bins inside it. Use the wrapper below so the correct Python and EULA
setting are applied automatically:

```bash
./scripts/run_conveyor_isaac.sh --config configs/conveyor_demo.json --headless
```

To open Isaac Sim in the recommended stable warehouse overview:

```bash
./scripts/run_conveyor_isaac.sh --config configs/conveyor_demo.json --viewport overview
```

To view the full large-facility scene with plastic detections and robot takeouts:

```bash
./scripts/run_conveyor_isaac.sh --config configs/recycling_facility_large_v2.json --viewport overview --detections
```

To launch the belt-focused `edco_conveyor_segment_a` mini-scene:

```bash
./scripts/run_conveyor_isaac.sh --config configs/edco_conveyor_segment_a.json --viewport overview
```

To look through the new overhead conveyor camera:

```bash
./scripts/run_conveyor_isaac.sh --config configs/edco_conveyor_segment_a.json --viewport overhead
```

To run that segment continuously in short clean replay cycles:

```bash
./scripts/run_conveyor_isaac.sh --config configs/edco_conveyor_segment_a.json --viewport overview --loop
```

To export RGB camera frames from the configured overhead camera:

```bash
./scripts/run_conveyor_isaac.sh --config configs/conveyor_demo.json --headless
```

If the interactive renderer feels unstable on your machine, use the lightest GUI path:

```bash
./scripts/run_conveyor_isaac.sh --config configs/conveyor_demo.json --bare-bones
```

Expected outputs:

- RGB frames in the configured `output_dir/frames`
- Episode metadata in `output_dir/episode_plan.json`

The bundled demo config models a single-stream inbound mix with these item families:

- Cardboard and mixed paper
- PET and HDPE containers
- Aseptic cartons and detergent jugs
- Aluminum and steel cans
- Film plastic
- Glass bottles and shard clusters
- Residue/contamination

The main line contains four ordered stations:

- `presort`
- `occ_screen`
- `magnet`
- `manual_qc`

The reconstruction-backed mini-scene is anchored to `data/inspection/belt_a.jpg`,
`data/edco_escondido_area_catwalk`, and `colmap/20260311_area_catwalk_seq/sparse/0`.
See `data/edco_conveyor_segment_a_anchor.md`.

## Current simplifications

- The facility shell comes from Isaac Sim's `Simple Warehouse`; recycling equipment is still overlaid procedurally.
- The main belt is a static mesh; belt motion is approximated by continuously applying
  forward velocity while items remain on the mainline.
- Station routing is rule-based and deterministic rather than physics-driven separation.
- Capture areas are side zones, not downstream branch conveyors.
- Physics still uses simple box/cylinder proxies, but visuals are dressed with bottle/can/carton/bag details.
- Interactive sensor-camera viewing is intentionally deferred; use the stable overview interactively and `--headless` for camera output.
- The large `recycling_facility_large_v2` scene now includes scripted CV detections and robot pick arms for PET, HDPE, and plastic film.
- Learned robot policy loading is not wired in yet; robot cells currently use the scripted pick controller path.
