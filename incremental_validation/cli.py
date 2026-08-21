from __future__ import annotations

from pathlib import Path
from typing import Sequence
import argparse
import json

import numpy as np

from .comparison import (
    IncrementalThresholds,
    calibrate_thresholds,
    compare_shared_logits,
    save_comparisons,
)


def _load_logits(path: str | Path):
    payload = np.load(Path(path), allow_pickle=False)
    required = {"categories", "presence_probabilities", "type_logits"}
    missing = required - set(payload.files)
    if missing:
        raise ValueError(f"shared-logit NPZ is missing keys: {sorted(missing)}")
    categories = payload["categories"].astype(str)
    presence = payload["presence_probabilities"]
    logits = payload["type_logits"]
    archive_ids = payload["archive_ids"].astype(str) if "archive_ids" in payload.files else None
    return categories, presence, logits, archive_ids


def command_calibrate(args: argparse.Namespace) -> None:
    categories, presence, logits, _ = _load_logits(args.validation_logits)
    thresholds, summary = calibrate_thresholds(
        categories,
        presence,
        logits,
        known_acceptance=args.known_acceptance,
        energy_temperature=args.energy_temperature,
    )
    thresholds.save(args.output)
    summary_path = Path(args.output).with_name(Path(args.output).stem + "_summary.json")
    summary_path.write_text(
        json.dumps(
            {
                "experiment_name": "基于作者盲识别核心的增量功能验证",
                "thresholds": thresholds.__dict__,
                **summary,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"thresholds": thresholds.__dict__, **summary}, indent=2))


def command_compare(args: argparse.Namespace) -> None:
    categories, presence, logits, archive_ids = _load_logits(args.test_logits)
    if presence.ndim != 3 or logits.shape != presence.shape + (4,):
        raise ValueError("test arrays must have [N,M,q] and [N,M,q,4]")
    thresholds = IncrementalThresholds.load(args.thresholds)
    comparisons = []
    for index, category in enumerate(categories):
        archive_id = index if archive_ids is None else str(archive_ids[index])
        comparisons.append(
            compare_shared_logits(
                str(category), presence[index], logits[index], thresholds, archive_id=archive_id
            )
        )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    save_comparisons(
        comparisons,
        output / "shared_logits_predictions.csv",
        output / "shared_logits_comparison.json",
    )
    print(f"wrote {output / 'shared_logits_comparison.json'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Shared-logit incremental open-set validation around the author Transformer"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    calibrate = subparsers.add_parser("calibrate")
    calibrate.add_argument("--validation-logits", required=True)
    calibrate.add_argument("--output", required=True)
    calibrate.add_argument("--known-acceptance", type=float, default=0.95)
    calibrate.add_argument("--energy-temperature", type=float, default=1.0)
    calibrate.set_defaults(func=command_calibrate)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--test-logits", required=True)
    compare.add_argument("--thresholds", required=True)
    compare.add_argument("--output", required=True)
    compare.set_defaults(func=command_compare)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
