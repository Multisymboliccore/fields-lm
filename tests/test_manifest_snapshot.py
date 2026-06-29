import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_bundled_frozen_snapshot_matches_manifest() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "fields_official" / "reference"
    vendor = root / "official_source"
    manifest = json.loads((vendor / "OFFICIAL_SOURCE_MANIFEST.json").read_text())
    rows = manifest["files"]
    assert len(rows) >= 20
    for row in rows:
        path = vendor / row["path"]
        assert path.is_file(), path
        assert _sha256(path) == row["sha256"], path
