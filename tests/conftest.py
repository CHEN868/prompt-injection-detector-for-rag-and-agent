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
        probability = 0.86 if "obsolete" in lowered or "concealed setup" in lowered else 0.05
        return TransformerPrediction(probability, self.model_status, self.model_name_or_path)


class FakeRiskModel:
    model_status = "xgboost_loaded"
    metadata = {"feature_schema_version": FEATURE_SCHEMA_VERSION, "metrics": {}}

    def predict_proba(self, features):
        probability = float(features["transformer_prob"])
        if features["is_external_source"]:
            probability += 0.04
        return min(1.0, probability)


@pytest.fixture
def model_runtime():
    return ModelRuntime(transformer=FakeTransformer(), risk_model=FakeRiskModel())
