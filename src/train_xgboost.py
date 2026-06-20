"""Train an XGBoost risk fusion model from feature CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .xgboost_risk_model import train_xgboost_model


FEATURE_NAMES = [
    "transformer_prob",
    "rule_block",
    "rule_score",
    "matched_rule_count",
    "has_instruction_override",
    "has_system_prompt_leakage",
    "has_tool_hijack",
    "source_trust",
    "permission_level_value",
    "history_risk_count",
    "is_external_source",
    "is_tool_output",
]


def label_to_int(value: str) -> int:
    normalized = value.strip().lower()
    if normalized in {"0", "normal", "allow", "safe"}:
        return 0
    if normalized in {"1", "injection", "review", "block", "risky"}:
        return 1
    raise ValueError(f"Unsupported final_label: {value}")


def train_from_csv(input_path: Path, output_path: Path) -> None:
    rows = []
    labels = []
    with input_path.open("r", encoding="utf-8", newline="") as source_file:
        reader = csv.DictReader(source_file)
        for row in reader:
            rows.append(row)
            labels.append(label_to_int(row["final_label"]))

    if not rows:
        raise ValueError("No feature rows found.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = train_xgboost_model(rows=rows, labels=labels, feature_names=FEATURE_NAMES)
    model.save_model(str(output_path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    train_from_csv(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
