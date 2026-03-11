from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


def export_visualization_bundle(
    episode_plan_path: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    plan_path = Path(episode_plan_path)
    payload = json.loads(plan_path.read_text())
    destination = Path(output_dir) if output_dir is not None else plan_path.parent / "viz"
    destination.mkdir(parents=True, exist_ok=True)

    files = {
        "summary": destination / "summary.md",
        "system_flow": destination / "system_flow.mmd",
        "recycling_flow": destination / "recycling_flow.mmd",
        "stream_counts": destination / "stream_counts.json",
        "stage_counts": destination / "stage_counts.json",
        "overview_html": destination / "overview.html",
    }

    system_flow = _build_system_flow_mermaid(payload)
    recycling_flow = _build_recycling_flow_mermaid(payload)
    files["summary"].write_text(_build_summary_markdown(payload))
    files["system_flow"].write_text(system_flow)
    files["recycling_flow"].write_text(recycling_flow)
    files["stream_counts"].write_text(json.dumps(payload.get("summaries", {}), indent=2))
    files["stage_counts"].write_text(json.dumps(_build_stage_counts(payload), indent=2))
    files["overview_html"].write_text(_build_overview_html(payload, system_flow, recycling_flow))
    return files


def _build_summary_markdown(payload: dict) -> str:
    lines = [
        "# Episode Plan Visualization Bundle",
        "",
        f"- Seed: `{payload.get('seed')}`",
        f"- Duration: `{payload.get('duration')}` seconds",
        f"- Recycling spawn events: `{len(payload.get('events', []))}`",
        f"- Total material units: `{len(payload.get('material_units', []))}`",
        "",
        "## Stream Summary",
        "",
        "| Stream | Generated | Delivered | Recovered/Terminal | Residual |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for stream, summary in sorted(payload.get("summaries", {}).items()):
        lines.append(
            f"| {stream} | {summary.get('generated_count', 0)} | {summary.get('delivered_count', 0)} | "
            f"{summary.get('captured_commodity_count', 0)} | {summary.get('residual_count', 0)} |"
        )

    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `system_flow.mmd`: end-to-end stream/facility Sankey-style Mermaid graph",
            "- `recycling_flow.mmd`: recycling-stage Mermaid graph with station and outcome counts",
            "- `stream_counts.json`: stream-level summary counts",
            "- `stage_counts.json`: lifecycle stage frequency counts by stream",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_system_flow_mermaid(payload: dict) -> str:
    source_counts = Counter(unit["source_id"] for unit in payload.get("material_units", []))
    stream_counts = {
        stream: summary.get("generated_count", 0)
        for stream, summary in payload.get("summaries", {}).items()
    }
    by_stream_facility = defaultdict(Counter)
    by_stream_terminal = defaultdict(Counter)

    for unit in payload.get("material_units", []):
        delivered = next((stage for stage in unit["lifecycle"] if stage["stage"] == "delivered_to_facility"), None)
        if delivered is not None:
            by_stream_facility[unit["stream"]][delivered["facility_id"]] += 1
        by_stream_terminal[unit["stream"]][unit["final_status"]] += 1

    lines = [
        "flowchart LR",
        "  classDef source fill:#d9ead3,stroke:#6aa84f,color:#000;",
        "  classDef stream fill:#cfe2f3,stroke:#3d85c6,color:#000;",
        "  classDef facility fill:#fce5cd,stroke:#e69138,color:#000;",
        "  classDef terminal fill:#ead1dc,stroke:#a64d79,color:#000;",
    ]

    for source_id, count in sorted(source_counts.items()):
        lines.append(f'  { _token(source_id) }["{source_id}\\n{count}"]:::source')
    for stream, count in sorted(stream_counts.items()):
        lines.append(f'  { _token(stream) }["{stream}\\n{count}"]:::stream')

    facility_ids = set()
    for counters in by_stream_facility.values():
        facility_ids.update(counters.keys())
    for facility_id in sorted(facility_ids):
        lines.append(f'  { _token(facility_id) }["{facility_id}"]:::facility')

    terminal_ids = set()
    for counters in by_stream_terminal.values():
        terminal_ids.update(counters.keys())
    for terminal in sorted(terminal_ids):
        lines.append(f'  { _token("terminal_" + terminal) }["{terminal}"]:::terminal')

    for source_id, count in sorted(source_counts.items()):
        lines.append(f"  {_token(source_id)} -->|{count}| {_token(_dominant_stream_for_source(payload, source_id))}")

    for stream, facilities in sorted(by_stream_facility.items()):
        for facility_id, count in sorted(facilities.items()):
            lines.append(f"  {_token(stream)} -->|{count}| {_token(facility_id)}")

    for stream, terminals in sorted(by_stream_terminal.items()):
        for terminal, count in sorted(terminals.items()):
            lines.append(f"  {_token(stream)} -->|{count}| {_token('terminal_' + terminal)}")

    return "\n".join(lines) + "\n"


def _build_recycling_flow_mermaid(payload: dict) -> str:
    stage_edges = Counter()
    final_counts = Counter()

    for lifecycle in payload.get("item_lifecycles", []):
        stages = lifecycle.get("lifecycle", [])
        for left, right in zip(stages, stages[1:]):
            stage_edges[(left["stage"], right["stage"])] += 1
        final_counts[lifecycle.get("final_status", "unknown")] += 1

    lines = [
        "flowchart LR",
        "  classDef stage fill:#d9d2e9,stroke:#674ea7,color:#000;",
        "  classDef outcome fill:#f4cccc,stroke:#cc0000,color:#000;",
    ]

    stage_names = sorted({name for edge in stage_edges for name in edge})
    for stage in stage_names:
        lines.append(f'  {_token(stage)}["{stage}"]:::stage')
    for outcome in sorted(final_counts):
        lines.append(f'  {_token("outcome_" + outcome)}["{outcome}\\n{final_counts[outcome]}"]:::outcome')

    for (left, right), count in sorted(stage_edges.items()):
        lines.append(f"  {_token(left)} -->|{count}| {_token(right)}")
    if stage_names:
        last_stage = max(stage_names, key=lambda name: sum(count for (left, _), count in stage_edges.items() if left == name))
        for outcome, count in sorted(final_counts.items()):
            lines.append(f"  {_token(last_stage)} -. {count} .-> {_token('outcome_' + outcome)}")
    return "\n".join(lines) + "\n"


def _build_stage_counts(payload: dict) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for unit in payload.get("material_units", []):
        for stage in unit.get("lifecycle", []):
            counts[unit["stream"]][stage["stage"]] += 1
    return {stream: dict(counter) for stream, counter in counts.items()}


def _dominant_stream_for_source(payload: dict, source_id: str) -> str:
    counts = Counter(
        unit["stream"] for unit in payload.get("material_units", []) if unit["source_id"] == source_id
    )
    if not counts:
        return "recycling"
    return counts.most_common(1)[0][0]


def _token(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


def _build_overview_html(payload: dict, system_flow: str, recycling_flow: str) -> str:
    stream_cards = []
    for stream, summary in sorted(payload.get("summaries", {}).items()):
        stream_cards.append(
            f"""
            <article class="card">
              <h3>{stream}</h3>
              <dl>
                <div><dt>Generated</dt><dd>{summary.get("generated_count", 0)}</dd></div>
                <div><dt>Delivered</dt><dd>{summary.get("delivered_count", 0)}</dd></div>
                <div><dt>Recovered</dt><dd>{summary.get("captured_commodity_count", 0)}</dd></div>
                <div><dt>Residual</dt><dd>{summary.get("residual_count", 0)}</dd></div>
              </dl>
            </article>
            """
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Episode Plan Overview</title>
  <style>
    :root {{
      --bg: #f5f1e8;
      --panel: #fffdf8;
      --ink: #1d241f;
      --muted: #5b675f;
      --line: #d7cfbf;
      --accent: #2f6b4f;
      --accent-2: #b86a2f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Iowan Old Style", "Palatino Linotype", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(47,107,79,0.12), transparent 28%),
        linear-gradient(180deg, #f7f3ea 0%, var(--bg) 100%);
    }}
    main {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    h1, h2, h3 {{ margin: 0 0 12px; }}
    h1 {{ font-size: clamp(2rem, 4vw, 3.4rem); line-height: 0.95; }}
    h2 {{ font-size: 1.2rem; letter-spacing: 0.04em; text-transform: uppercase; color: var(--accent); }}
    p {{ color: var(--muted); max-width: 70ch; }}
    .hero {{
      display: grid;
      gap: 20px;
      padding: 28px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.75);
      backdrop-filter: blur(6px);
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
      margin: 24px 0 30px;
    }}
    .card, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      padding: 18px;
      box-shadow: 0 8px 24px rgba(0,0,0,0.04);
    }}
    dl {{
      margin: 0;
      display: grid;
      gap: 10px;
    }}
    dl div {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding-top: 8px;
      border-top: 1px solid var(--line);
    }}
    dt {{ color: var(--muted); }}
    dd {{ margin: 0; font-weight: 700; }}
    .panels {{
      display: grid;
      gap: 18px;
    }}
    .mermaid {{
      overflow-x: auto;
    }}
    pre {{
      white-space: pre-wrap;
      font-size: 0.85rem;
      color: var(--muted);
      background: #f3eee2;
      padding: 14px;
      border: 1px solid var(--line);
      overflow-x: auto;
    }}
    .meta {{
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 0.95rem;
    }}
    .meta strong {{ color: var(--accent-2); }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>EDCO Material Flow Overview</h1>
      <div class="meta">
        <span><strong>Seed</strong> {payload.get("seed")}</span>
        <span><strong>Duration</strong> {payload.get("duration")}s</span>
        <span><strong>Material Units</strong> {len(payload.get("material_units", []))}</span>
        <span><strong>MRF Spawn Events</strong> {len(payload.get("events", []))}</span>
      </div>
      <p>Generated headlessly from <code>episode_plan.json</code>. Open this file in a browser to inspect the end-to-end system flow and the recycling-stage flow as rendered diagrams.</p>
    </section>

    <section>
      <div class="cards">
        {"".join(stream_cards)}
      </div>
    </section>

    <section class="panels">
      <article class="panel">
        <h2>System Flow</h2>
        <div class="mermaid">
{system_flow}
        </div>
      </article>

      <article class="panel">
        <h2>Recycling Flow</h2>
        <div class="mermaid">
{recycling_flow}
        </div>
      </article>

      <article class="panel">
        <h2>Fallback</h2>
        <p>If Mermaid does not render in your browser, the raw diagram source is included below.</p>
        <pre>{system_flow}\n\n{recycling_flow}</pre>
      </article>
    </section>
  </main>
  <script type="module">
    import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
    mermaid.initialize({{
      startOnLoad: true,
      theme: "base",
      themeVariables: {{
        primaryColor: "#e9efe8",
        primaryTextColor: "#1d241f",
        primaryBorderColor: "#2f6b4f",
        lineColor: "#6b7c71",
        tertiaryColor: "#f6f1e8",
        fontFamily: 'Georgia, "Palatino Linotype", serif'
      }}
    }});
  </script>
</body>
</html>
"""
