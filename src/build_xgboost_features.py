"""Build XGBoost feature CSV from a context dataset CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .context_feature_builder import features_for_csv, build_chunk_features
from .context_models import ContextChunk
from .context_rule_detector import detect_context_rules
from .transformer_predictor import predict_chunk


def build_features(input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8", newline="") as source_file:
        reader = csv.DictReader(source_file)
        rows = []
        for row in reader:
            chunk = ContextChunk(
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
            rule_result = detect_context_rules(chunk.content)
            transformer_prediction = predict_chunk(chunk)
            features = features_for_csv(
                build_chunk_features(
                    chunk=chunk,
                    rule_result=rule_result,
                    transformer_prob=transformer_prediction.transformer_prob,
                )
            )
            rows.append(
                {
                    "id": row["id"],
                    "case_id": row.get("case_id", ""),
                    **features,
                    "final_label": row.get("final_label", row.get("label", "")),
                }
            )

    if not rows:
        raise ValueError("No rows found in input dataset.")

    with output_path.open("w", encoding="utf-8", newline="") as target_file:
        writer = csv.DictWriter(target_file, fieldnames=list(rows[0].keys()))
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
