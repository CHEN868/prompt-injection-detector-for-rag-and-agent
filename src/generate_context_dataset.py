"""Generate a small illustrative context_dataset.csv from built-in demos."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .context_scanner import flatten_context_request, scan_context
from .demo_context_cases import get_demo_cases


def generate_dataset(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in get_demo_cases():
        result = scan_context(case.request)
        result_by_chunk = {item.chunk_id: item for item in result.chunk_results}
        for chunk in flatten_context_request(case.request):
            chunk_result = result_by_chunk[chunk.chunk_id]
            rows.append(
                {
                    "id": chunk.chunk_id,
                    "case_id": case.case_id,
                    "text": chunk.content,
                    "final_label": "normal" if chunk_result.decision == "ALLOW" else "injection",
                    "scenario": chunk.scenario,
                    "context_role": chunk.context_role,
                    "source": chunk.source,
                    "source_trust": chunk.source_trust,
                    "permission_level": chunk.permission_level or "none",
                    "history_risk_count": chunk.history_risk_count,
                    "notes": "illustrative demo row, not production training data",
                }
            )

    with output_path.open("w", encoding="utf-8", newline="") as target_file:
        writer = csv.DictWriter(target_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/context_dataset.csv")
    args = parser.parse_args()
    generate_dataset(Path(args.output))


if __name__ == "__main__":
    main()
