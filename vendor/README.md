# Pinned candidate-set source

The candidate-set classifier is loaded from the upstream repository at:

- repository: `https://github.com/zhouph0313/DNA`
- branch: `main`
- commit: `1ac47fce3bb9526633e38a7863612d7dc5db3a40`

The recorded commit does not contain a license file. For that reason, the source
snapshot is not redistributed in this pre-publication package. After permission
has been confirmed, clone the exact commit into `vendor/zhouph0313_DNA` and run:

```bash
python -m author_baseline.cli verify-vendor
```

`zhouph0313_DNA.snapshot.json` records the SHA-256 value of every expected file.
