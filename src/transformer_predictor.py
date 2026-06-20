"""Optional Hugging Face Transformer predictor for chunk-level text risk."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_BACKBONE = "distilbert-base-multilingual-cased"
DEFAULT_LOCAL_MODEL_DIR = Path("models/transformer_prompt_injection")


@dataclass
class TransformerPrediction:
    transformer_prob: float
    model_status: str
    model_name_or_path: str | None = None


class TransformerPredictor:
    """Thin wrapper around AutoModelForSequenceClassification.

    When no model is explicitly loaded, this class returns a not_configured
    prediction instead of pretending to have a calibrated classifier.
    """

    def __init__(
        self,
        tokenizer: Any | None = None,
        model: Any | None = None,
        model_status: str = "not_configured",
        model_name_or_path: str | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.model_status = model_status
        self.model_name_or_path = model_name_or_path

    def predict_proba(self, text: str) -> float:
        if self.tokenizer is None or self.model is None:
            return 0.0

        try:
            import torch
        except ImportError:
            self.model_status = "missing_dependency"
            return 0.0

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
        return float(probabilities[0, 1].item())

    def predict_chunk(self, chunk: Any) -> TransformerPrediction:
        return TransformerPrediction(
            transformer_prob=self.predict_proba(chunk.content),
            model_status=self.model_status,
            model_name_or_path=self.model_name_or_path,
        )


def load_model(model_path_or_name: str | None = None) -> TransformerPredictor:
    """Load a local fine-tuned model or a Hugging Face backbone.

    If a Hugging Face backbone is loaded without local fine-tuning artifacts, the
    model_status is untrained_backbone and its probabilities are not calibrated
    for this project.
    """
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        return TransformerPredictor(model_status="missing_dependency")

    requested = model_path_or_name
    local_model = DEFAULT_LOCAL_MODEL_DIR
    if requested is None and local_model.exists():
        requested = str(local_model)
    if requested is None:
        requested = DEFAULT_BACKBONE

    is_local_model = Path(requested).exists()
    status = "fine_tuned" if is_local_model else "untrained_backbone"
    tokenizer = AutoTokenizer.from_pretrained(requested)
    model = AutoModelForSequenceClassification.from_pretrained(
        requested,
        num_labels=2,
        id2label={0: "normal", 1: "injection"},
        label2id={"normal": 0, "injection": 1},
    )
    return TransformerPredictor(
        tokenizer=tokenizer,
        model=model,
        model_status=status,
        model_name_or_path=requested,
    )


def predict_proba(text: str, predictor: TransformerPredictor | None = None) -> float:
    if predictor is None:
        return 0.0
    return predictor.predict_proba(text)


def predict_chunk(chunk: Any, predictor: TransformerPredictor | None = None) -> TransformerPrediction:
    if predictor is None:
        return TransformerPrediction(transformer_prob=0.0, model_status="not_configured")
    return predictor.predict_chunk(chunk)
