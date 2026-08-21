# Open-Set Blind ECC Recognition from DNA Sequencing Reads

This repository is the reproducibility package for a hierarchical framework that
recognizes error-correcting-code (ECC) metadata directly from variable-length DNA
sequencing reads affected by insertion, deletion, and substitution errors.

The inference workflow is:

1. coded/uncoded discrimination;
2. post-hoc detection of code types outside the candidate set;
3. candidate-set code-type recognition among BCH, convolutional, LDPC, and
   Polar codes; and
4. conditional code-rate and codeword-length recognition for LDPC inputs.

Four neural recognition backbones are represented in the supplied checkpoints:
CNN, LSTM, Transformer, and ResNet. Reads use four-channel nucleotide one-hot
representations, explicit validity masks, and probability-level decision fusion
across reads and DNA molecules.

> **Release status:** this is a pre-publication staging package. Complete the
> items in `PREPUBLICATION_CHECKLIST.md` before making the repository public.

## Repository contents

- `author_baseline/`: adapters around the fixed candidate-set classifier and
  probability-level fusion.
- `incremental_validation/`: coded/uncoded, post-hoc OOD, robustness, and
  five-seed evaluation workflows.
- `hierarchical_ecc/`: simulation and data-generation utilities used by the
  evaluation workflow. This directory is not the original candidate-set
  recognizer.
- `artifacts/model_weights/`: CNN, LSTM, Transformer, and ResNet checkpoints for
  code-type, LDPC code-rate, and LDPC codeword-length recognition.
- `artifacts/presence/`: fitted coded/uncoded discriminator and threshold.
- `artifacts/ood/`: serialized post-hoc OOD detector, feature definition, and
  fixed detector configuration.
- `results/reference/`: compact reference results, including the five
  independently generated test datasets with seeds 46--50.
- `third_party/`: metadata and adapters for external code families.
- `vendor/`: the pinned upstream-source manifest. The unlicensed upstream source
  is deliberately not redistributed in this staging package.
- `docs/reproduction.md`: environment, verification, smoke-test, and formal-run
  instructions.

## Reported reference results

Under `p_ins = p_del = p_sub = 0.05`, `M = 20` molecules, and `q = 50` reads per
molecule, the pooled results over five independently generated test datasets are:

- archive-level candidate-set code-type accuracy: 81.5%;
- specificity for uncoded inputs: 100%;
- detection rate for code types outside the candidate set: 90.5%; and
- acceptance rate for candidate-set code types: 97.9%.

The machine-readable source is
`results/reference/stage7_five_seed/aggregate_metrics.json`.

## Quick verification

After installing the environment and obtaining the pinned candidate-set source:

```bash
python -m author_baseline.cli verify-vendor
python -m author_baseline.cli inspect-weights --weights-root artifacts/model_weights --device cpu
python -m pytest -q tests
```

See `docs/reproduction.md` for the complete commands and important limitations.

## Data and long-term archival

The GitHub package contains compact predictions, metrics, configurations, and
model artifacts. The full reference sequences, per-read predictions, and dataset
manifests are intended for a DOI-issuing data repository. Replace the placeholders
in `DATA_AVAILABILITY.md` before publication.

## Citation

Citation metadata are provided in `CITATION.cff`. Author names, repository URL,
and the paper DOI must be completed before the public release.

## License

No project license has yet been selected. Public redistribution must wait until
the project license and the status of the pinned upstream source have been
resolved; see `LICENSE_PENDING.md`.
