# Reproduction guide

## 1. Environment

The frozen formal evaluation used Python 3.10, NumPy 2.2.6, PyTorch 2.5.1
with CUDA 12.1, and an NVIDIA GeForce RTX 3050 Laptop GPU. A CPU can run the
verification and smoke tests, but the full five-seed experiment is substantially
more expensive.

Using Conda:

```bash
conda env create -f environment.yml
conda activate dna-ecc-blind-recognition
```

Alternatively, create a Python 3.10 environment and install
`requirements.txt`. Install `requirements-optional.txt` only for the legacy data
scripts and plotting utilities that require those packages.

## 2. Obtain the pinned candidate-set source

The recorded upstream commit has no license file, so its source is not copied
into this staging package. After resolving permission, obtain the exact commit:

```bash
git clone https://github.com/zhouph0313/DNA vendor/zhouph0313_DNA
git -C vendor/zhouph0313_DNA checkout 1ac47fce3bb9526633e38a7863612d7dc5db3a40
```

Verify the commit and every recorded file:

```bash
python -m author_baseline.cli verify-vendor
```

## 3. Verify model artifacts

```bash
python -m author_baseline.cli inspect-weights \
  --weights-root artifacts/model_weights \
  --device cpu
```

The command must report SHA-256 matches and strict state-dictionary loading for
all 12 checkpoints.

The fitted gating artifacts are:

- `artifacts/presence/models/external_presence_cnn.pt`;
- `artifacts/presence/thresholds.json`;
- `artifacts/ood/structural_embedding_proxy_detector.npz`;
- `artifacts/ood/frozen_detector_config.json`; and
- `artifacts/ood/feature_definitions.json`.

## 4. Run tests

```bash
python -m pytest -q tests
```

The original formal environment recorded 86 passing tests after the final
Stage-7 regression tests were added. Results can differ if tests requiring
external inner-code executables have not been built.

## 5. Run a small smoke experiment

First audit the fixed inputs and thresholds:

```bash
python -m incremental_validation.stage7_repeatability audit \
  --output examples/smoke_test/output \
  --source artifacts/presence \
  --stage5 artifacts/ood \
  --stage6 results/reference/stage6_robustness \
  --device cpu
```

Then run the smoke dataset:

```bash
python -m incremental_validation.stage7_repeatability smoke \
  --output examples/smoke_test/output \
  --source artifacts/presence \
  --stage5 artifacts/ood \
  --stage6 results/reference/stage6_robustness \
  --device cpu \
  --workers 1 \
  --batch-size 16
```

HEDGES and DNA-Aeon generation additionally requires the upstream projects and
the command-line adapters described in `third_party/adapters/README.md`.

## 6. Reproduce the formal five-seed evaluation

The formal settings are recorded in
`results/reference/stage7_five_seed/config.json`. The run uses seeds 46--50,
seven input categories, 100 archives per category and seed, 20 molecules per
archive, and 50 reads per molecule.

For each seed:

```bash
python -m incremental_validation.stage7_repeatability seed \
  --seed 46 \
  --output runs/stage7_five_seed \
  --source artifacts/presence \
  --stage5 artifacts/ood \
  --stage6 results/reference/stage6_robustness \
  --device cuda \
  --workers 8 \
  --batch-size 96 \
  --resume
```

Repeat with seeds 47, 48, 49, and 50, then run the `summarize` and `finalize`
modes with the same paths. Compare the resulting metrics against
`results/reference/stage7_five_seed/aggregate_metrics.json`.

## 7. Statistical unit and confidence intervals

The archive is the statistical unit. The confidence intervals use a hierarchical
cluster bootstrap: resample test-data seeds first, then resample archives within
each seed and true input category. Reads from the same molecule are not treated
as independent observations.

## 8. Known limitation in this staging package

All rate and length checkpoints are included, but the current public-facing
adapter constructs only the code-type Transformer. Before release, add a unified
entry point that loads and evaluates the conditional LDPC code-rate and
codeword-length models using the exact test manifests reported in the paper.
