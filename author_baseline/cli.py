from __future__ import annotations

from pathlib import Path
from typing import Sequence
import argparse
import json
import subprocess
import sys

import torch

from .original_models import AUTHOR_MODEL_NAMES, create_original_model
from .vendor_guard import DEFAULT_VENDOR_ROOT, verify_vendor_snapshot
from .weights import DEFAULT_WEIGHT_ROOT, inspect_author_weights


def command_verify(args: argparse.Namespace) -> None:
    result = verify_vendor_snapshot(args.repository)
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    if not result.valid:
        raise SystemExit(1)


def command_smoke(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    batch, length = 2, 400
    indices = torch.arange(length) % 4
    one_hot = torch.nn.functional.one_hot(indices, num_classes=4).T.float()
    x = one_hot.unsqueeze(0).repeat(batch, 1, 1).to(device)
    mask = torch.ones((batch, length), dtype=torch.float32, device=device)
    results: list[dict[str, object]] = []
    for name in AUTHOR_MODEL_NAMES:
        model = create_original_model(name, num_classes=4, repository=args.repository).to(device).eval()
        with torch.inference_mode():
            logits = model(x, mask)
        results.append(
            {
                "model": name,
                "class": model.__class__.__name__,
                "input_shape": list(x.shape),
                "output_shape": list(logits.shape),
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
            }
        )
    print(json.dumps({"vendor_verified": True, "models": results}, indent=2))


def command_inspect_weights(args: argparse.Namespace) -> None:
    records = inspect_author_weights(args.weights_root, args.device)
    result = {
        "weight_root": str(Path(args.weights_root).resolve()),
        "count": len(records),
        "all_hashes_match": len(records) == 12 and all(r.hash_matches for r in records),
        "all_strict_load_compatible": len(records) == 12
        and all(r.strict_load_compatible for r in records),
        "primary_type_checkpoint": "type/transformer_model_f10.6033.pt",
        "parameter_models_connected_this_round": False,
        "records": [record.to_dict() for record in records],
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["all_hashes_match"] or not result["all_strict_load_compatible"]:
        raise SystemExit(1)


def command_author_vote(args: argparse.Namespace) -> None:
    verification = verify_vendor_snapshot(args.repository)
    if not verification.valid:
        raise RuntimeError("refusing to run a modified author repository")
    script = Path(args.repository) / "vote6.1copyerror.py"
    command = [
        sys.executable,
        str(script),
        "--csv",
        str(Path(args.csv).resolve()),
        "--model",
        f"{args.model_name}:{Path(args.checkpoint).resolve()}",
        "--output",
        str(Path(args.output).resolve()),
        "--seed",
        str(args.seed),
        "--num_copies",
        str(args.num_copies),
        "--error_rate",
        str(args.error_rate),
        "--group_size",
        str(args.group_size),
        "--output-interval",
        str(args.output_interval),
    ]
    if args.filter_range is not None:
        command.extend(
            [
                "--filter_column",
                args.filter_column,
                "--filter_range",
                str(args.filter_range[0]),
                str(args.filter_range[1]),
            ]
        )
    # The original script is executed as a subprocess from its immutable path.
    subprocess.run(command, cwd=str(Path(args.repository)), check=True)


def command_author_generator(args: argparse.Namespace) -> None:
    verification = verify_vendor_snapshot(args.repository)
    if not verification.valid:
        raise RuntimeError("refusing to run a modified author repository")
    script_names = {
        "BCH": "bchDNA4.py",
        "Convolutional": "conveDNA4.py",
        "LDPC": "ldpcDNA4.py",
        "Polar": "polarDNA4.py",
    }
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    script = Path(args.repository) / script_names[args.code_type]
    # Author generators use relative OUTPUT_DIR values. Selecting cwd adapts
    # the destination without changing a line of the original script.
    subprocess.run([sys.executable, str(script)], cwd=str(output_root), check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Adapters for the unmodified author baseline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser("verify-vendor")
    verify.add_argument("--repository", default=str(DEFAULT_VENDOR_ROOT))
    verify.set_defaults(func=command_verify)

    smoke = subparsers.add_parser("smoke-models")
    smoke.add_argument("--repository", default=str(DEFAULT_VENDOR_ROOT))
    smoke.add_argument("--device", default="cpu")
    smoke.set_defaults(func=command_smoke)

    weights = subparsers.add_parser("inspect-weights")
    weights.add_argument("--weights-root", default=str(DEFAULT_WEIGHT_ROOT))
    weights.add_argument("--device", default="cpu")
    weights.set_defaults(func=command_inspect_weights)

    vote = subparsers.add_parser("run-author-vote")
    vote.add_argument("--repository", default=str(DEFAULT_VENDOR_ROOT))
    vote.add_argument("--csv", required=True)
    vote.add_argument("--model-name", choices=AUTHOR_MODEL_NAMES, required=True)
    vote.add_argument("--checkpoint", required=True)
    vote.add_argument("--output", required=True)
    vote.add_argument("--seed", type=int, default=42)
    vote.add_argument("--num-copies", type=int, default=100)
    vote.add_argument("--error-rate", type=float, default=0.05)
    vote.add_argument("--group-size", type=int, default=5)
    vote.add_argument("--output-interval", type=int, default=500)
    vote.add_argument("--filter-column", default="sample_idx")
    vote.add_argument("--filter-range", type=int, nargs=2)
    vote.set_defaults(func=command_author_vote)

    generator = subparsers.add_parser("run-author-generator")
    generator.add_argument("--repository", default=str(DEFAULT_VENDOR_ROOT))
    generator.add_argument(
        "--code-type", choices=("BCH", "Convolutional", "LDPC", "Polar"), required=True
    )
    generator.add_argument("--output-root", required=True)
    generator.set_defaults(func=command_author_generator)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
