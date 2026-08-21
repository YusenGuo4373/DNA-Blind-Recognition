# Package manifest

This staging package contains the compact materials needed to prepare a public
reproducibility repository.

## Included

- project documentation, citation template, environment specifications, and
  data-availability template;
- candidate-set classifier adapters and probability-level fusion;
- coded/uncoded and post-hoc OOD evaluation code;
- simulation, IDS-channel, robustness, ablation, and five-seed evaluation code;
- 12 CNN/LSTM/Transformer/ResNet checkpoints for code type, LDPC code rate, and
  LDPC codeword length;
- fitted coded/uncoded and OOD detector artifacts;
- compact Stage-6 robustness results;
- compact Stage-7 results for independently generated test-data seeds 46--50;
- per-archive predictions, confusion matrices, and hierarchical cluster
  bootstrap intervals; and
- third-party provenance records and source adapters.

## Deliberately excluded

- approximately 385 MB of Stage-7 per-read shards and reference-sequence
  intermediates;
- training and calibration feature caches;
- redundant experiment-stage directories and failed intermediate runs;
- cross-simulator experiments that are not part of the manuscript's main
  evaluation;
- the unlicensed pinned candidate-set source snapshot;
- compiled third-party binaries;
- manuscript build artifacts, presentations, review material, and local caches.

## Integrity

`manifests/artifact_sha256.json` records a relative path, byte size, and SHA-256
digest for every staged file except the checksum manifest itself.
