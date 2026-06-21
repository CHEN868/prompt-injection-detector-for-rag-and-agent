"""Pipeline step: context_dataset.csv -> model signals -> xgboost_features.csv."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .context_feature_builder import FEATURE_NAMES, build_chunk_features, features_for_csv
from .context_models import ContextChunk
from .context_rule_detector import detect_context_rules
from .transformer_predictor import TransformerPredictor, load_model


def build_features(
    input_path: Path,
    output_path: Path,
    predictor: TransformerPredictor | None = None,
) -> None:
    with input_path.open("r", encoding="utf-8", newline="") as source_file:
        source_rows = list(csv.DictReader(source_file))
    if not source_rows:
        raise ValueError("No rows found in input dataset.")

    chunks = [
        ContextChunk(
            chunk_id=row["id"],
            case_id=row.get("case_id") or None,
            scenario=row.get("scenario") or "chat",
            context_role=row.get("context_role") or "user_input",
            content=row["text"],
            source=row.get("source") or "unknown",
            source_trust=float(row.get("source_trust") or 0.5),
            permission_level=row.get("permission_level") or "none",
            history_risk_count=int(row.get("history_risk_count") or 0),
        )
        for row in source_rows
    ]
    active_predictor = predictor or load_model(local_files_only=True)
    probabilities = active_predictor.predict_many(chunk.content for chunk in chunks)

    rows = []
    for row, chunk, transformer_prob in zip(source_rows, chunks, probabilities, strict=True):
        rule_result = detect_context_rules(chunk.content)
        features = features_for_csv(
            build_chunk_features(
                chunk=chunk,
                rule_result=rule_result,
                transformer_prob=transformer_prob,
            )
        )
        rows.append(
            {
                "id": row["id"],
                "group_id": row.get("group_id", row["id"]),
                "case_id": row.get("case_id", ""),
                "split": row.get("split", "train"),
                **features,
                "final_label": row.get("final_label", row.get("label", "")),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "group_id", "case_id", "split", *FEATURE_NAMES, "final_label"]
    with output_path.open("w", encoding="utf-8", newline="") as target_file:
        writer = csv.DictWriter(target_file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    build_features(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
