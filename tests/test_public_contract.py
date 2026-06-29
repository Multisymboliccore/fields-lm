from fields_official import OFFICIAL_ARCHITECTURE, FieldsConfig


def test_promoted_topology_contract() -> None:
    assert OFFICIAL_ARCHITECTURE["block_count"] == 24
    assert len(OFFICIAL_ARCHITECTURE["native_field_positions"]) == 18
    assert OFFICIAL_ARCHITECTURE["mamba2_positions"] == (10, 22)
    assert OFFICIAL_ARCHITECTURE["refresh_positions"] == (5, 11, 17, 23)
    assert FieldsConfig().pcaf_enabled is True
