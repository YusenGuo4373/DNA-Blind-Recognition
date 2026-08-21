# Staging-package validation report

Validation date: 2026-08-21

The following checks were completed after assembling the package:

- Python AST syntax parsing passed for all 48 staged Python files.
- JSON parsing passed for all 25 JSON files present at validation time.
- The four headline values in the root README were checked directly against
  `results/reference/stage7_five_seed/aggregate_metrics.json`.
- The staged Transformer, coded/uncoded discriminator, and post-hoc OOD detector
  match the SHA-256 values recorded by the formal experiment.
- The complete relative-path checksum manifest had zero missing files and zero
  checksum mismatches.
- Local absolute-path patterns and common secret/key patterns produced zero
  matches after the cross-simulator script was excluded.

The full PyTorch test suite was not rerun inside the packaging runtime because
that runtime does not provide the paper's PyTorch environment. The formal
experiment audit records 86 passing project tests. The public release should be
tested again from a clean Python 3.10 environment after the pinned upstream
source has been obtained.
