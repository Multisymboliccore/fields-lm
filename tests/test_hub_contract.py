from fields_official import FieldsHubModel


def test_hub_class_contract() -> None:
    assert FieldsHubModel.CONFIG_FILENAME == "config.json"
    assert FieldsHubModel.WEIGHTS_FILENAME == "model.safetensors"
