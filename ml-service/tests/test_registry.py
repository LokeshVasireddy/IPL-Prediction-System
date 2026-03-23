from core.model_loader import load_production_model


def test_production_model_loads():
    bundle, metadata = load_production_model()

    assert bundle is not None
    assert metadata["model_name"] is not None
