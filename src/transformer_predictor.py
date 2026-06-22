"""Hugging Face predictor for multilingual prompt-injection detection."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_MODEL_ID = "Verm1ion/injection-sentry-xlmr"
DEFAULT_MODEL_REVISION = "936bd6aeed82a583fd28e733184ab62e028715ea"
DEFAULT_MAX_LENGTH = 512
DEFAULT_STRIDE = 128
LOGGER = logging.getLogger(__name__)


class TransformerLoadError(RuntimeError):
    """Raised when the configured semantic detector cannot be loaded."""


@dataclass(frozen=True)
class TransformerPrediction:
    transformer_prob: float
    model_status: str
    model_name_or_path: str


def _select_device(torch: Any) -> str:
    requested = os.getenv("PROMPT_GUARD_DEVICE")
    if requested:
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _find_injection_label(model: Any) -> int:
    id2label = {int(key): str(value).lower() for key, value in model.config.id2label.items()}
    for label_id, label in id2label.items():
        if any(marker in label for marker in ("injection", "malicious", "attack", "unsafe")):
            return label_id
    raise TransformerLoadError(
        f"Cannot identify injection label from model id2label={model.config.id2label!r}."
    )


class TransformerPredictor:
    """Reusable model wrapper with sliding-window and batch inference."""

    model_status = "transformer_loaded"

    def __init__(
        self,
        tokenizer: Any,
        model: Any,
        model_name_or_path: str,
        revision: str | None,
        device: str,
        max_length: int = DEFAULT_MAX_LENGTH,
        stride: int = DEFAULT_STRIDE,
        batch_size: int = 8,
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.model_name_or_path = model_name_or_path
        self.revision = revision
        self.device = device
        self.max_length = max_length
        self.stride = stride
        self.batch_size = batch_size
        self.injection_label_id = _find_injection_label(model)

    def _window_probabilities(self, texts: list[str]) -> list[float]:
        import torch

        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            stride=self.stride,
            return_overflowing_tokens=True,
        )
        mapping = encoded.pop("overflow_to_sample_mapping").tolist()
        encoded.pop("num_truncated_tokens", None)
        maxima = [0.0] * len(texts)
        total_windows = len(mapping)
        for start in range(0, total_windows, self.batch_size):
            end = min(start + self.batch_size, total_windows)
            batch = {key: value[start:end].to(self.device) for key, value in encoded.items()}
            with torch.inference_mode():
                logits = self.model(**batch).logits
                probabilities = torch.softmax(logits, dim=-1)[:, self.injection_label_id]
            for offset, probability in enumerate(probabilities.detach().cpu().tolist()):
                sample_index = mapping[start + offset]
                maxima[sample_index] = max(maxima[sample_index], float(probability))
        return maxima

    def predict_many(self, texts: Iterable[str]) -> list[float]:
        values = [text.strip() for text in texts]
        if not values or any(not text for text in values):
            raise ValueError("Transformer input text must not be empty.")
        return self._window_probabilities(values)

    def predict_proba(self, text: str) -> float:
        print("Running Transformer predict_proba")
        LOGGER.debug("Running Transformer predict_proba")
        return self.predict_many([text])[0]

    def predict_chunk(self, chunk: Any) -> TransformerPrediction:
        print("Running Transformer predict_chunk")
        LOGGER.debug("Running Transformer predict_chunk")
        return TransformerPrediction(
            transformer_prob=self.predict_proba(chunk.content),
            model_status=self.model_status,
            model_name_or_path=self.model_name_or_path,
        )


def load_model(
    model_path_or_name: str | None = None,
    *,
    revision: str | None = None,
    local_files_only: bool = True,
) -> TransformerPredictor:
    """Load the configured fine-tuned classifier or raise a clear error."""
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as error:
        raise TransformerLoadError("Install transformers and torch before loading the model.") from error

    requested = model_path_or_name or os.getenv("PROMPT_GUARD_MODEL_ID", DEFAULT_MODEL_ID)
    configured_revision = revision or os.getenv("PROMPT_GUARD_MODEL_REVISION", DEFAULT_MODEL_REVISION)
    if Path(requested).exists():
        configured_revision = None

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            requested,
            revision=configured_revision,
            local_files_only=local_files_only,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            requested,
            revision=configured_revision,
            local_files_only=local_files_only,
        )
    except Exception as error:
        mode = "local cache" if local_files_only else "Hugging Face Hub"
        raise TransformerLoadError(f"Cannot load {requested!r} from {mode}: {error}") from error

    device = _select_device(torch)
    model.to(device)
    model.eval()
    return TransformerPredictor(
        tokenizer=tokenizer,
        model=model,
        model_name_or_path=requested,
        revision=configured_revision,
        device=device,
    )


def predict_proba(text: str, predictor: TransformerPredictor) -> float:
    LOGGER.debug("Running Transformer predict_proba")
    return predictor.predict_proba(text)


def predict_chunk(chunk: Any, predictor: TransformerPredictor) -> TransformerPrediction:
    LOGGER.debug("Running Transformer predict_chunk")
    return predictor.predict_chunk(chunk)
