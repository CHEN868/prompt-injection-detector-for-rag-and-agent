from types import SimpleNamespace

import pytest

from src.context_feature_builder import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from src.xgboost_risk_model import RiskModelLoadError, _validate_metadata


def test_xgboost_metadata_rejects_wrong_feature_order():
    model = SimpleNamespace(n_features_in_=len(FEATURE_NAMES))
    metadata = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": list(reversed(FEATURE_NAMES)),
    }

    with pytest.raises(RiskModelLoadError, match="feature names/order"):
        _validate_metadata(metadata, model)
