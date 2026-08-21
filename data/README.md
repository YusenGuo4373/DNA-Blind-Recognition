# Data package

No large generated dataset is committed to this GitHub staging directory.

The compact repository contains:

- exact formal settings and independent test-data seeds;
- data-independence audits;
- per-archive predictions;
- aggregate and per-seed metrics; and
- fixed model and detector artifacts.

The complete DOI-archived data package should contain:

1. generated reference sequences for every formal archive and molecule;
2. archive, molecule, and read identifiers;
3. train, validation, calibration, and test split manifests;
4. encoder-family, code-rate, codeword-length, and encoder-instance provenance;
5. IDS-channel settings and read-level random seeds;
6. per-read logits, embeddings or probabilities needed by the published
   evaluation; and
7. a SHA-256 manifest covering every deposited file.

The five-seed formal evaluation represents 3,500 archives, 70,000 unique
reference molecules, and 3,500,000 reads. Keep these data in a DOI-issuing data
repository rather than ordinary Git history.

Dataset DOI: `[TO BE ADDED]`
