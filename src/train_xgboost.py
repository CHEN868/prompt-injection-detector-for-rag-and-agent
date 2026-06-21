"""Train and evaluate the real XGBoost context-risk fusion artifact."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from .context_feature_builder import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from .transformer_predictor import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION
from .xgboost_risk_model import train_xgboost_model


def label_to_int(value: str) -> int:
    normalized = value.strip().lower()
    if normalized in {"0", "normal", "allow", "safe"}:
        return 0
    if normalized in {"1", "injection", "review", "block", "risky"}:
        return 1
    raise ValueError(f"Unsupported final_label: {value}")


def _metrics(model: Any, rows: list[dict[str, str]]) -> dict[str, float | int]:
    if not rows:
        return {"samples": 0}
    matrix = [[float(row[name]) for name in FEATURE_NAMES] for row in rows]
    labels = [label_to_int(row["final_label"]) for row in rows]
    probabilities = model.predict_proba(matrix)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    result: dict[str, float | int] = {
        "samples": len(rows),
        "accuracy": round(float(accuracy_score(labels, predictions)), 4),
        "precision": round(float(precision_score(labels, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(labels, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(labels, predictions, zero_division=0)), 4),
    }
    if len(set(labels)) == 2:
        result["roc_auc"] = round(float(roc_auc_score(labels, probabilities)), 4)
    return result


def train_from_csv(
    input_path: Path,
    output_path: Path,
    metadata_path: Path | None = None,
) -> dict[str, Any]:
    with input_path.open("r", encoding="utf-8", newline="") as source_file:
        rows = list(csv.DictReader(source_file))
    if not rows:
        raise ValueError("No feature rows found.")

    train_rows = [row for row in rows if row.get("split", "train") == "train"]
    validation_rows = [row for row in rows if row.get("split") == "validation"]
    test_rows = [row for row in rows if row.get("split") == "test"]
    labels = [label_to_int(row["final_label"]) for row in train_rows]
    model = train_xgboost_model(train_rows, labels, FEATURE_NAMES)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output_path))
    metadata_file = metadata_path or output_path.with_name(f"{output_path.stem}.meta.json")
    metadata = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": FEATURE_NAMES,
        "transformer_model_id": DEFAULT_MODEL_ID,
        "transformer_model_revision": DEFAULT_MODEL_REVISION,
        "training_data_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metrics": {
            "validation": _metrics(model, validation_rows),
            "test": _metrics(model, test_rows),
        },
        "sample_counts": {
            "train": len(train_rows),
            "validation": len(validation_rows),
            "test": len(test_rows),
        },
        "limitations": "Deterministic demo data; metrics are not production claims.",
    }
    metadata_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata")
    args = parser.parse_args()
    metadata = train_from_csv(
        Path(args.input),
        Path(args.output),
        Path(args.metadata) if args.metadata else None,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
