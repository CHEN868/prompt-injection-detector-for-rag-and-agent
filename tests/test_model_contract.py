from types import SimpleNamespace

import pytest

from src.context_feature_builder import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    validate_feature_contract,
)
from src.xgboost_risk_model import RiskModelLoadError, _validate_metadata


def test_xgboost_metadata_rejects_wrong_feature_order():
    model = SimpleNamespace(n_features_in_=len(FEATURE_NAMES))
    metadata = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": list(reversed(FEATURE_NAMES)),
    }

    with pytest.raises(RiskModelLoadError, match="feature names/order"):
        _validate_metadata(metadata, model)


def test_v3_feature_contract_is_small_and_has_no_rule_flags():
    validate_feature_contract(FEATURE_NAMES)

    assert len(FEATURE_NAMES) == 5
    assert FEATURE_NAMES == [
        "transformer_prob",
        "rule_score",
        "context_risk_score",
        "source_trust_encoded",
        "permission_level_encoded",
    ]

    with pytest.raises(ValueError, match="Forbidden"):
        validate_feature_contract(["transformer_prob", "rule_block"])
