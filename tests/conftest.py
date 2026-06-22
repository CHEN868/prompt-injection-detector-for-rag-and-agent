from types import SimpleNamespace

import pytest

from src.context_feature_builder import FEATURE_SCHEMA_VERSION
from src.model_runtime import ModelRuntime
from src.transformer_predictor import TransformerPrediction


class FakeTransformer:
    model_status = "transformer_loaded"
    model_name_or_path = "test-transformer"
    revision = "test-revision"
    device = "cpu"

    def predict_chunk(self, chunk):
        lowered = chunk.content.lower()
        risky_markers = (
            "obsolete",
            "concealed setup",
            "忽略之前",
            "send_email",
            "api key",
            "attacker@example.com",
            "隐藏提示词",
        )
        probability = 0.9 if any(marker in lowered for marker in risky_markers) else 0.05
        return TransformerPrediction(probability, self.model_status, self.model_name_or_path)


class FakeRiskModel:
    model_status = "xgboost_loaded"
    metadata = {"feature_schema_version": FEATURE_SCHEMA_VERSION, "metrics": {}}

    def predict_proba(self, features):
        assert set(features) == {
            "transformer_prob",
            "rule_score",
            "context_risk_score",
            "source_trust_encoded",
            "permission_level_encoded",
        }
        return min(
            1.0,
            float(features["transformer_prob"]) * 0.8
            + float(features["context_risk_score"]) * 0.2,
        )


@pytest.fixture
def model_runtime():
    return ModelRuntime(transformer=FakeTransformer(), risk_model=FakeRiskModel())
