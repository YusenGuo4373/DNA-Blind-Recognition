from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Sequence
import argparse
import json

import torch

from .config import ExperimentConfig
from .evaluation import aggregate_seed_metrics, calibrate_thresholds, evaluate_seed
from .models import load_checkpoint
from .training import train_model
from .validation import validation_report


def smoke_config(config: ExperimentConfig) -> ExperimentConfig:
    """Small end-to-end configuration; model architecture stays unchanged."""

    return replace(
        config,
        voting=replace(
            config.voting,
            default_molecules=2,
            default_reads=2,
            molecule_sweep=(1, 2),
            read_sweep=(1, 2),
        ),
        training=replace(
            config.training,
            batch_size=8,
            presence_epochs=1,
            type_epochs=1,
            train_known_reads_per_type=16,
            train_no_ecc_reads_per_subtype=32,
            val_known_reads_per_type=8,
            val_no_ecc_reads_per_subtype=16,
            calibration_archives_per_category=2,
            test_archives_per_category=1,
            early_stopping_patience=1,
        ),
    )


def _config(path: str | None, smoke: bool) -> ExperimentConfig:
    config = ExperimentConfig.load(path) if path else ExperimentConfig()
    return smoke_config(config) if smoke else config


def _seeds(config: ExperimentConfig, requested: Sequence[int] | None, smoke: bool) -> tuple[int, ...]:
    if requested:
        return tuple(int(value) for value in requested)
    return (config.training.seeds[0],) if smoke else config.training.seeds


def _device(requested: str | None) -> str:
    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def _models(output: Path, seed: int, device: str):
    presence, presence_meta = load_checkpoint(
        output / "models" / f"presence_seed_{seed}.pt", device=device
    )
    type_model, type_meta = load_checkpoint(
        output / "models" / f"type_seed_{seed}.pt", device=device
    )
    if presence_meta["task"] != "presence" or type_meta["task"] != "type":
        raise ValueError("checkpoint task metadata do not match the cascade stages")
    return presence, type_model


def command_validate(args: argparse.Namespace) -> None:
    config = _config(args.config, smoke=False)
    report = validation_report(config)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


def command_init_config(args: argparse.Namespace) -> None:
    config = ExperimentConfig()
    config.save(args.output)
    print(f"wrote {args.output}")


def command_train(args: argparse.Namespace) -> None:
    config = _config(args.config, args.smoke)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config.save(output / "resolved_config.json")
    seeds = _seeds(config, args.seeds, args.smoke)
    tasks = ("presence", "type") if args.task == "all" else (args.task,)
    for seed in seeds:
        for task in tasks:
            print(f"training task={task} seed={seed}", flush=True)
            result = train_model(
                config,
                task=task,
                seed=seed,
                output_directory=output / "models",
                device=_device(args.device),
            )
            print(
                f"saved {result.checkpoint}; best_epoch={result.best_epoch}; "
                f"val_macro_f1={result.best_validation_macro_f1:.4f}",
                flush=True,
            )


def command_calibrate(args: argparse.Namespace) -> None:
    config = _config(args.config, args.smoke)
    output = Path(args.output)
    seeds = _seeds(config, args.seeds, args.smoke)
    device = _device(args.device)
    for seed in seeds:
        print(f"calibrating seed={seed} (known/no-ECC validation only)", flush=True)
        presence, type_model = _models(output, seed, device)
        thresholds = calibrate_thresholds(
            config,
            seed,
            presence,
            type_model,
            output / "calibration",
            device=device,
            batch_size=args.inference_batch_size,
        )
        print(json.dumps({"seed": seed, **thresholds.__dict__}, ensure_ascii=False), flush=True)


def command_evaluate(args: argparse.Namespace) -> None:
    config = _config(args.config, args.smoke)
    output = Path(args.output)
    seeds = _seeds(config, args.seeds, args.smoke)
    device = _device(args.device)
    metric_paths: list[Path] = []
    errors = (0.0, 0.10, 0.20) if args.smoke else None
    for seed in seeds:
        print(f"evaluating seed={seed}; fountain remains test-only", flush=True)
        presence, type_model = _models(output, seed, device)
        from .voting import Thresholds

        thresholds = Thresholds.load(output / "calibration" / f"thresholds_seed_{seed}.json")
        summary = evaluate_seed(
            config,
            seed,
            presence,
            type_model,
            thresholds,
            output / "evaluation",
            device=device,
            batch_size=args.inference_batch_size,
            test_error_rates=errors,
        )
        metric_paths.append(output / "evaluation" / f"metrics_seed_{seed}.json")
        print(json.dumps({"seed": seed, "feasibility": summary["feasibility"]}, ensure_ascii=False))
    if len(metric_paths) > 1:
        aggregate_seed_metrics(metric_paths, output / "evaluation" / "metrics_aggregate.json")


def command_run(args: argparse.Namespace) -> None:
    command_train(args)
    command_calibrate(args)
    command_evaluate(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paper-compatible hierarchical ECC/no-ECC/unknown-ECC experiment"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init-config", help="write the default JSON configuration")
    init_parser.add_argument("--output", default="hierarchical_ecc_config.json")
    init_parser.set_defaults(func=command_init_config)

    validate_parser = subparsers.add_parser("validate", help="validate encoders and data semantics")
    validate_parser.add_argument("--config")
    validate_parser.set_defaults(func=command_validate)

    def common(run_parser: argparse.ArgumentParser) -> None:
        run_parser.add_argument("--config", help="JSON config; defaults to the protocol values")
        run_parser.add_argument("--output", default="outputs/hierarchical_ecc")
        run_parser.add_argument("--seeds", nargs="+", type=int)
        run_parser.add_argument("--device", help="for example cuda, cuda:0, or cpu")
        run_parser.add_argument("--smoke", action="store_true", help="small pipeline validation run")
        run_parser.add_argument("--inference-batch-size", type=int, default=256)

    train_parser = subparsers.add_parser("train", help="train the two independent read classifiers")
    common(train_parser)
    train_parser.add_argument("--task", choices=("presence", "type", "all"), default="all")
    train_parser.set_defaults(func=command_train)

    calibrate_parser = subparsers.add_parser("calibrate", help="select tau1 and tau2 on validation")
    common(calibrate_parser)
    calibrate_parser.set_defaults(func=command_calibrate)

    evaluate_parser = subparsers.add_parser("evaluate", help="evaluate held-out known/no-ECC/LT archives")
    common(evaluate_parser)
    evaluate_parser.set_defaults(func=command_evaluate)

    run_parser = subparsers.add_parser("run", help="train, calibrate, and evaluate")
    common(run_parser)
    run_parser.add_argument("--task", choices=("all",), default="all", help=argparse.SUPPRESS)
    run_parser.set_defaults(func=command_run)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
