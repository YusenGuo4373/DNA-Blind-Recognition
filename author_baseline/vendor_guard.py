from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VENDOR_ROOT = WORKSPACE_ROOT / "vendor" / "zhouph0313_DNA"
DEFAULT_SNAPSHOT = WORKSPACE_ROOT / "vendor" / "zhouph0313_DNA.snapshot.json"


@dataclass(frozen=True)
class VendorVerification:
    valid: bool
    expected_commit: str
    actual_commit: str | None
    changed: tuple[str, ...]
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "expected_commit": self.expected_commit,
            "actual_commit": self.actual_commit,
            "changed": list(self.changed),
            "missing": list(self.missing),
            "unexpected": list(self.unexpected),
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _head_commit(repository: Path) -> str | None:
    head_path = repository / ".git" / "HEAD"
    if not head_path.exists():
        return None
    head = head_path.read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head
    reference = head.removeprefix("ref: ")
    loose = repository / ".git" / reference
    if loose.exists():
        return loose.read_text(encoding="utf-8").strip()
    packed = repository / ".git" / "packed-refs"
    if packed.exists():
        for line in packed.read_text(encoding="utf-8").splitlines():
            if not line.startswith(("#", "^")):
                commit, name = line.split(" ", 1)
                if name == reference:
                    return commit
    return None


def verify_vendor_snapshot(
    repository: str | Path = DEFAULT_VENDOR_ROOT,
    snapshot_path: str | Path = DEFAULT_SNAPSHOT,
) -> VendorVerification:
    repository = Path(repository).resolve()
    snapshot = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
    expected_files: dict[str, str] = snapshot["files"]
    actual_files = {
        path.relative_to(repository).as_posix(): path
        for path in repository.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(repository).parts
    }
    missing = tuple(sorted(set(expected_files) - set(actual_files)))
    unexpected = tuple(sorted(set(actual_files) - set(expected_files)))
    changed = tuple(
        sorted(
            relative
            for relative in set(expected_files) & set(actual_files)
            if _sha256(actual_files[relative]) != expected_files[relative]
        )
    )
    expected_commit = str(snapshot["commit"])
    actual_commit = _head_commit(repository)
    return VendorVerification(
        valid=not changed and not missing and not unexpected and actual_commit == expected_commit,
        expected_commit=expected_commit,
        actual_commit=actual_commit,
        changed=changed,
        missing=missing,
        unexpected=unexpected,
    )


def require_clean_vendor(repository: str | Path = DEFAULT_VENDOR_ROOT) -> Path:
    verification = verify_vendor_snapshot(repository)
    if not verification.valid:
        raise RuntimeError(f"author vendor snapshot verification failed: {verification.to_dict()}")
    return Path(repository).resolve()
