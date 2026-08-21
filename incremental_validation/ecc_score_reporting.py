from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import json

import numpy as np

from .comparison import IncrementalThresholds, KNOWN_TYPES, NO_ECC_TYPES


SEVEN_GROUPS = KNOWN_TYPES + ("NoECC", "HEDGES", "DNA-Aeon")


def _group(category: str) -> str:
    return "NoECC" if category in NO_ECC_TYPES else category


def _summary(values: np.ndarray, threshold: float) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "min": float(values.min()),
        "q05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.50)),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
        "below_tau1_rate": float(np.mean(values < threshold)),
        "at_or_above_tau1_rate": float(np.mean(values >= threshold)),
    }


def _binary_auroc(positive: np.ndarray, negative: np.ndarray) -> float:
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    comparisons = (positive[:, None] > negative[None, :]).mean()
    ties = (positive[:, None] == negative[None, :]).mean()
    return float(comparisons + 0.5 * ties)


def write_ecc_score_distributions(
    shared_logits_npz: str | Path,
    thresholds: IncrementalThresholds,
    output_directory: str | Path,
) -> dict[str, Any]:
    payload = np.load(Path(shared_logits_npz), allow_pickle=False)
    categories = payload["categories"].astype(str)
    archive_ids = payload["archive_ids"].astype(str)
    read_probabilities = np.asarray(payload["presence_probabilities"], dtype=np.float64)
    if read_probabilities.ndim != 3:
        raise ValueError("presence probabilities must have shape [N,M,q]")
    scores = read_probabilities.mean(axis=2).mean(axis=1)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)

    with (output / "ecc_score_by_archive.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("archive_id", "category", "seven_group", "ecc_score", "tau1", "stage1_output"),
        )
        writer.writeheader()
        for archive_id, category, score in zip(archive_ids, categories, scores):
            writer.writerow(
                {
                    "archive_id": archive_id,
                    "category": category,
                    "seven_group": _group(category),
                    "ecc_score": float(score),
                    "tau1": thresholds.ecc_presence,
                    "stage1_output": "ecc" if score >= thresholds.ecc_presence else "no_ecc",
                }
            )

    category_summary = {
        category: _summary(scores[categories == category], thresholds.ecc_presence)
        for category in sorted(set(categories))
    }
    groups = np.asarray([_group(value) for value in categories], dtype=str)
    seven_summary = {
        group: _summary(scores[groups == group], thresholds.ecc_presence)
        for group in SEVEN_GROUPS
        if np.any(groups == group)
    }
    no_ecc_scores = scores[np.isin(categories, NO_ECC_TYPES)]
    diagnostic = {
        "tau1": thresholds.ecc_presence,
        "score_definition": "mean_q_then_mean_M_of_per_read_ECC_probability",
        "category_count": int(len(category_summary)),
        "seven_group_count": int(len(seven_summary)),
        "category_distribution": category_summary,
        "seven_group_distribution": seven_summary,
        "stage1_supervised_inner_code_vs_no_ecc_auroc": {
            category: _binary_auroc(scores[categories == category], no_ecc_scores)
            for category in ("HEDGES", "DNA-Aeon")
        },
        "diagnostic_rule": (
            "HEDGES and DNA-Aeon are supervised ECC-positive classes at stage one. They pass "
            "the ECC gate when archive ecc_score is at least tau1; their final unknown_ecc "
            "status is evaluated only by the external stage-two energy gate."
        ),
        "protocol": "stage1_supervised_inner_codes_stage2_open_set_rejection",
    }
    (output / "ecc_score_distribution.json").write_text(
        json.dumps(diagnostic, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for filename, rows in (
        ("ecc_score_distribution_8_categories.csv", category_summary),
        ("ecc_score_distribution_7_groups.csv", seven_summary),
    ):
        with (output / filename).open("w", newline="", encoding="utf-8") as handle:
            first = next(iter(rows.values()))
            writer = csv.DictWriter(handle, fieldnames=("category", *first.keys()))
            writer.writeheader()
            for category, values in rows.items():
                writer.writerow({"category": category, **values})

    width, height = 900, 550
    left, right, top, bottom = 70, 25, 50, 70
    plot_width = width - left - right
    plot_height = height - top - bottom
    palette = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#7f7f7f", "#9467bd", "#17becf")

    def xy(x_value: float, y_value: float) -> tuple[float, float]:
        return left + x_value * plot_width, top + (1.0 - y_value) * plot_height

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="25" text-anchor="middle" font-family="sans-serif" font-size="17">Stage-one ECC-score distribution (M/q soft voting)</text>',
    ]
    for tick in np.linspace(0.0, 1.0, 6):
        x_tick, y_zero = xy(float(tick), 0.0)
        _, y_tick = xy(0.0, float(tick))
        svg.append(f'<line x1="{x_tick:.2f}" y1="{top}" x2="{x_tick:.2f}" y2="{top+plot_height}" stroke="#dddddd"/>')
        svg.append(f'<line x1="{left}" y1="{y_tick:.2f}" x2="{left+plot_width}" y2="{y_tick:.2f}" stroke="#dddddd"/>')
        svg.append(f'<text x="{x_tick:.2f}" y="{top+plot_height+22}" text-anchor="middle" font-family="sans-serif" font-size="11">{tick:.1f}</text>')
        svg.append(f'<text x="{left-12}" y="{y_tick+4:.2f}" text-anchor="end" font-family="sans-serif" font-size="11">{tick:.1f}</text>')
    svg.append(f'<line x1="{left}" y1="{top+plot_height}" x2="{left+plot_width}" y2="{top+plot_height}" stroke="black"/>')
    svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_height}" stroke="black"/>')
    tau_x, _ = xy(thresholds.ecc_presence, 0.0)
    svg.append(f'<line x1="{tau_x:.2f}" y1="{top}" x2="{tau_x:.2f}" y2="{top+plot_height}" stroke="black" stroke-dasharray="6,5"/>')
    svg.append(f'<text x="{tau_x+5:.2f}" y="{top+14}" font-family="sans-serif" font-size="11">tau1={thresholds.ecc_presence:.4f}</text>')
    for group_index, group in enumerate(SEVEN_GROUPS):
        values = np.sort(scores[groups == group])
        if not values.size:
            continue
        points: list[tuple[float, float]] = [(0.0, 0.0)]
        previous_y = 0.0
        for index, value in enumerate(values, start=1):
            next_y = index / values.size
            points.append((float(value), previous_y))
            points.append((float(value), next_y))
            previous_y = next_y
        points.append((1.0, 1.0))
        encoded_points = " ".join(f"{xy(x, y)[0]:.2f},{xy(x, y)[1]:.2f}" for x, y in points)
        color = palette[group_index]
        svg.append(f'<polyline points="{encoded_points}" fill="none" stroke="{color}" stroke-width="2"/>')
        legend_x = left + 15 + (group_index % 4) * 190
        legend_y = height - 38 + (group_index // 4) * 18
        svg.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x+22}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        svg.append(f'<text x="{legend_x+28}" y="{legend_y+4}" font-family="sans-serif" font-size="11">{group}</text>')
    svg.append(f'<text x="{left+plot_width/2}" y="{height-8}" text-anchor="middle" font-family="sans-serif" font-size="13">Archive ECC score</text>')
    svg.append(f'<text x="18" y="{top+plot_height/2}" text-anchor="middle" transform="rotate(-90 18 {top+plot_height/2})" font-family="sans-serif" font-size="13">Empirical CDF</text>')
    svg.append("</svg>")
    (output / "ecc_score_ecdf.svg").write_text("\n".join(svg), encoding="utf-8")
    return diagnostic
