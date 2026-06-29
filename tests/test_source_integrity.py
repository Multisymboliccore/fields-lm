from pathlib import Path

import pytest

from fields_official import FieldsConfig, verify_frozen_source


def test_source_snapshot_when_canonical_is_present() -> None:
    config = FieldsConfig()
    canonical = config.resolved_canonical_source()
    if not Path(canonical).is_file():
        pytest.skip("canonical file is installed by assemble_release_on_h100.sh")
    audit = verify_frozen_source(config)
    assert audit["canonical_sha256"]
    assert len(audit["verified_snapshot_files"]) >= 20
