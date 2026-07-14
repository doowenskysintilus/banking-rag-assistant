"""Turns an `EvalReport` into human-facing output: a Markdown report (written to
disk automatically after every run — see `backend.evaluation.runner`) and a
self-contained HTML dashboard (no external CDN/script — served directly by
`backend.routers.evaluation`, safe under a strict CSP).

Both renderers apply the same "higher is better" color thresholds uniformly to
every metric (see `backend.evaluation.types` on why `hallucination_rate` is
stored inverted) — mirrors the high/medium/low bands `backend.rag.confidence`
already uses elsewhere in this app, so a "green" score means the same thing
across the whole product.
"""

from __future__ import annotations

import html
from datetime import datetime

from backend.evaluation.types import EvalReport, METRIC_NAMES

_HIGH_THRESHOLD = 0.75
_MEDIUM_THRESHOLD = 0.4

_METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "context_precision": "Context Precision",
    "context_recall": "Context Recall",
    "answer_relevancy": "Answer Relevancy",
    "hallucination_rate": "Non-Hallucination Rate",
}


def _band(score: float) -> str:
    if score >= _HIGH_THRESHOLD:
        return "good"
    if score >= _MEDIUM_THRESHOLD:
        return "warning"
    return "critical"


def _band_color(band: str) -> str:
    return {"good": "#1a9e5c", "warning": "#c98a12", "critical": "#c0392b"}[band]


# --- Markdown report (written to disk after every run) -----------------------


def format_markdown_report(report: EvalReport) -> str:
    lines = [
        f"# RAG Evaluation Report — {report.run_id}",
        "",
        f"- **Dataset**: {report.dataset_name}",
        f"- **Samples**: {report.num_samples}",
        f"- **Generated**: {report.created_at.isoformat()}",
        "",
        "## Aggregate scores",
        "",
        "| Metric | Score | Band | Provider |",
        "|---|---|---|---|",
    ]
    for metric in METRIC_NAMES:
        score = report.aggregate_scores.get(metric)
        provider = report.providers_used.get(metric, "—")
        if score is None:
            lines.append(f"| {_METRIC_LABELS[metric]} | n/a | skipped | {provider} |")
            continue
        lines.append(
            f"| {_METRIC_LABELS[metric]} | {score:.3f} | {_band(score)} | {provider} |"
        )

    if report.warnings:
        lines += ["", "## Warnings", ""]
        lines += [f"- {w}" for w in report.warnings]

    lines += [
        "",
        "## Per-sample scores",
        "",
        "| Sample | " + " | ".join(_METRIC_LABELS[m] for m in METRIC_NAMES) + " |",
    ]
    lines.append("|---" * (len(METRIC_NAMES) + 1) + "|")
    for sample_eval in report.sample_evaluations:
        cells = []
        for metric in METRIC_NAMES:
            score = sample_eval.score_for(metric)
            cells.append(f"{score:.2f}" if score is not None else "—")
        lines.append(f"| {sample_eval.sample.sample_id} | " + " | ".join(cells) + " |")

    return "\n".join(lines) + "\n"


# --- HTML dashboard ------------------------------------------------------------


def _sparkline_svg(values: list[float], width: int = 160, height: int = 36) -> str:
    """Tiny inline SVG trend line — no charting library, no external request."""
    if len(values) < 2:
        return ""
    step = width / (len(values) - 1)
    points = " ".join(
        f"{i * step:.1f},{height - v * height:.1f}" for i, v in enumerate(values)
    )
    last = values[-1]
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'class="sparkline" preserveAspectRatio="none">'
        f'<polyline points="{points}" fill="none" stroke="{_band_color(_band(last))}" stroke-width="2"/>'
        f"</svg>"
    )


def _metric_card_html(
    metric: str, score: float | None, history_scores: list[float]
) -> str:
    label = _METRIC_LABELS[metric]
    if score is None:
        return (
            f'<div class="card skipped"><div class="card-label">{html.escape(label)}</div>'
            f'<div class="card-score">n/a</div><div class="card-band">skipped — no provider available</div></div>'
        )
    band = _band(score)
    spark = _sparkline_svg(history_scores) if len(history_scores) >= 2 else ""
    return (
        f'<div class="card {band}"><div class="card-label">{html.escape(label)}</div>'
        f'<div class="card-score">{score:.2f}</div><div class="card-band">{band}</div>{spark}</div>'
    )


def _sample_row_html(sample_eval, metrics: tuple[str, ...]) -> str:
    cells = []
    for metric in metrics:
        score = sample_eval.score_for(metric)
        if score is None:
            cells.append('<td class="na">—</td>')
        else:
            cells.append(f'<td class="{_band(score)}">{score:.2f}</td>')
    question = html.escape(sample_eval.sample.question)
    return f'<tr><td class="question">{question}</td>' + "".join(cells) + "</tr>"


def render_dashboard_html(
    report: EvalReport,
    history: list[tuple[datetime, dict[str, float]]] | None = None,
) -> str:
    """`history` is `[(created_at, aggregate_scores), ...]` for prior runs of the
    same dataset, oldest first, used only to draw the per-metric trend sparkline.
    """
    history = history or []

    cards_html = "".join(
        _metric_card_html(
            metric,
            report.aggregate_scores.get(metric),
            [h[1][metric] for h in history if metric in h[1]]
            + (
                [report.aggregate_scores[metric]]
                if metric in report.aggregate_scores
                else []
            ),
        )
        for metric in METRIC_NAMES
    )

    rows_html = "".join(
        _sample_row_html(s, METRIC_NAMES) for s in report.sample_evaluations
    )
    header_cells = "".join(f"<th>{_METRIC_LABELS[m]}</th>" for m in METRIC_NAMES)

    warnings_html = ""
    if report.warnings:
        items = "".join(f"<li>{html.escape(w)}</li>" for w in report.warnings)
        warnings_html = (
            f'<div class="warnings"><strong>Warnings</strong><ul>{items}</ul></div>'
        )

    return f"""\
<title>RAG Evaluation Dashboard</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; padding: 2rem;
          background: #f7f8fa; color: #1a1a1a; }}
  @media (prefers-color-scheme: dark) {{ body {{ background: #14161a; color: #e8e8e8; }} }}
  :root[data-theme="dark"] body {{ background: #14161a; color: #e8e8e8; }}
  :root[data-theme="light"] body {{ background: #f7f8fa; color: #1a1a1a; }}
  h1 {{ font-size: 1.4rem; margin-bottom: 0.25rem; }}
  .meta {{ color: #6b7280; font-size: 0.85rem; margin-bottom: 1.5rem; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem;
            margin-bottom: 2rem; }}
  .card {{ border-radius: 10px; padding: 1rem; background: #fff; border: 1px solid #e5e7eb; }}
  @media (prefers-color-scheme: dark) {{ .card {{ background: #1f2228; border-color: #2c303a; }} }}
  :root[data-theme="dark"] .card {{ background: #1f2228; border-color: #2c303a; }}
  .card-label {{ font-size: 0.8rem; color: #6b7280; margin-bottom: 0.4rem; }}
  .card-score {{ font-size: 1.8rem; font-weight: 600; }}
  .card-band {{ font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.03em; margin-top: 0.2rem; }}
  .card.good .card-score, .card.good .card-band {{ color: #1a9e5c; }}
  .card.warning .card-score, .card.warning .card-band {{ color: #c98a12; }}
  .card.critical .card-score, .card.critical .card-band {{ color: #c0392b; }}
  .card.skipped {{ opacity: 0.6; }}
  table {{ border-collapse: collapse; width: 100%; overflow-x: auto; display: block; }}
  th, td {{ padding: 0.5rem 0.75rem; text-align: center; border-bottom: 1px solid #e5e7eb; font-size: 0.85rem; }}
  @media (prefers-color-scheme: dark) {{ th, td {{ border-color: #2c303a; }} }}
  :root[data-theme="dark"] th, :root[data-theme="dark"] td {{ border-color: #2c303a; }}
  td.question {{ text-align: left; max-width: 360px; }}
  td.good {{ color: #1a9e5c; font-weight: 600; }}
  td.warning {{ color: #c98a12; font-weight: 600; }}
  td.critical {{ color: #c0392b; font-weight: 600; }}
  td.na {{ color: #9ca3af; }}
  .warnings {{ margin: 1.5rem 0; padding: 0.75rem 1rem; border-radius: 8px; background: #fff7e6;
               border: 1px solid #f0d896; color: #7a5a00; }}
  @media (prefers-color-scheme: dark) {{ .warnings {{ background: #2a2410; border-color: #4a3d10; color: #e0c250; }} }}
</style>
<h1>RAG Evaluation Dashboard</h1>
<div class="meta">Run {html.escape(report.run_id)} · dataset "{html.escape(report.dataset_name)}" ·
  {report.num_samples} samples · {report.created_at.isoformat()}</div>
<div class="cards">{cards_html}</div>
{warnings_html}
<table>
  <thead><tr><th>Question</th>{header_cells}</tr></thead>
  <tbody>{rows_html}</tbody>
</table>
"""
