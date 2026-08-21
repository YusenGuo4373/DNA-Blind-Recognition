from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape
import math


CURVE_METRICS = (
    ("no_ecc_specificity", "No-ECC specificity", "#3366cc"),
    ("fountain_ecc_recall", "Fountain ECC recall", "#dc3912"),
    ("fountain_unknown_recall", "Fountain unknown recall", "#ff9900"),
    ("known_type_macro_f1", "Known type macro-F1", "#109618"),
    ("end_to_end_accuracy", "End-to-end accuracy", "#990099"),
)


def write_curve_svgs(rows: list[dict[str, object]], directory: str | Path, seed: int) -> list[Path]:
    """Write dependency-free SVG performance curves for error, q and M."""

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for curve in ("error_rate", "q", "M"):
        subset = sorted(
            (row for row in rows if row["curve"] == curve), key=lambda row: float(row["x"])
        )
        if not subset:
            continue
        path = directory / f"curve_{curve}_seed_{seed}.svg"
        path.write_text(_curve_svg(curve, subset), encoding="utf-8")
        paths.append(path)
    return paths


def _curve_svg(curve: str, rows: list[dict[str, object]]) -> str:
    width, height = 900, 520
    left, top, right, bottom = 90, 55, 250, 70
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_values = [float(row["x"]) for row in rows]
    x_min, x_max = min(x_values), max(x_values)

    def x_position(value: float) -> float:
        if x_max == x_min:
            return left + plot_width / 2
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def y_position(value: float) -> float:
        return top + (1.0 - max(0.0, min(1.0, value))) * plot_height

    title = {"error_rate": "Performance vs IDS probability", "q": "Performance vs reads per molecule (q)", "M": "Performance vs molecules (M)"}[curve]
    x_label = {"error_rate": "p_ins = p_del = p_sub", "q": "q", "M": "M"}[curve]
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="30" font-family="sans-serif" font-size="20" font-weight="bold">{escape(title)}</text>',
    ]
    for tick in np_ticks(0.0, 1.0, 0.2):
        y = y_position(tick)
        elements.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#dddddd"/>')
        elements.append(f'<text x="{left - 12}" y="{y + 5:.2f}" text-anchor="end" font-family="sans-serif" font-size="12">{tick:.1f}</text>')
    elements.extend(
        (
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="black"/>',
            f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="black"/>',
            f'<text x="25" y="{top + plot_height / 2}" transform="rotate(-90 25 {top + plot_height / 2})" text-anchor="middle" font-family="sans-serif" font-size="14">Metric</text>',
            f'<text x="{left + plot_width / 2}" y="{height - 20}" text-anchor="middle" font-family="sans-serif" font-size="14">{escape(x_label)}</text>',
        )
    )
    for value in x_values:
        x = x_position(value)
        label = f"{value:g}"
        elements.append(f'<line x1="{x:.2f}" y1="{top + plot_height}" x2="{x:.2f}" y2="{top + plot_height + 6}" stroke="black"/>')
        elements.append(f'<text x="{x:.2f}" y="{top + plot_height + 23}" text-anchor="middle" font-family="sans-serif" font-size="12">{label}</text>')

    for legend_index, (key, label, color) in enumerate(CURVE_METRICS):
        points: list[str] = []
        circles: list[str] = []
        for row in rows:
            value = float(row[key])
            if not math.isfinite(value):
                continue
            x = x_position(float(row["x"]))
            y = y_position(value)
            points.append(f"{x:.2f},{y:.2f}")
            circles.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5" fill="{color}"/>')
        if points:
            elements.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>')
            elements.extend(circles)
        legend_y = top + 15 + legend_index * 30
        legend_x = left + plot_width + 25
        elements.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 25}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        elements.append(f'<text x="{legend_x + 34}" y="{legend_y + 5}" font-family="sans-serif" font-size="12">{escape(label)}</text>')
    elements.append("</svg>")
    return "\n".join(elements)


def np_ticks(start: float, stop: float, step: float) -> list[float]:
    count = int(round((stop - start) / step))
    return [start + index * step for index in range(count + 1)]
