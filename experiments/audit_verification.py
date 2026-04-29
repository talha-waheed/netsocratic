"""
Audit verification-agent rule outputs for an experiment results folder.

Usage:
  python experiments/audit_verification.py results/experiments/1_10
  python experiments/audit_verification.py results/experiments/1_10 --output audit.csv

The script expects a results folder containing summary.json plus per-row pipeline
directories. For each row, it reports whether the selected winner rules match
the correct formal specification, and how many verification candidates produced
the same normalized rules.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_json_text(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _load_json_file(path: Path) -> Optional[Dict[str, Any]]:
    try:
        value = json.loads(path.read_text())
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _normalise_rules(spec: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Normalize a rules dict for stable equality comparisons."""
    if spec is None:
        return None

    try:
        reachability = spec.get("reachability") or {}
        waypoint = spec.get("waypoint") or {}
        loadbalancing = spec.get("loadbalancing") or {}

        return {
            "reachability": {
                str(src).lower(): sorted(str(prefix) for prefix in prefixes)
                for src, prefixes in reachability.items()
            },
            "waypoint": {
                str(key).lower(): sorted(str(node).lower() for node in nodes)
                for key, nodes in waypoint.items()
            },
            "loadbalancing": {
                str(key).lower(): int(count)
                for key, count in loadbalancing.items()
            },
        }
    except Exception:
        return None


def _canonical_rules(spec: Optional[Dict[str, Any]]) -> Optional[str]:
    normalized = _normalise_rules(spec)
    if normalized is None:
        return None
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def _find_pipeline_dir(
    results_dir: Path,
    row_id: str,
    summary_pipeline_dir: Optional[str],
) -> Optional[Path]:
    """Resolve the pipeline directory inside results_dir, tolerating stale summary paths."""
    candidates = []  # type: List[Path]

    if summary_pipeline_dir:
        pipeline_path = Path(summary_pipeline_dir)
        if pipeline_path.exists():
            candidates.append(pipeline_path)

        # Some copied/renamed result folders keep stale pipeline_dir values in
        # summary.json. Rebuild the path using the current results folder.
        parts = pipeline_path.parts
        if row_id in parts:
            row_index = parts.index(row_id)
            suffix = Path(*parts[row_index:])
            candidates.append(results_dir / suffix)

    row_root = results_dir / row_id
    if row_root.exists():
        for child in sorted(row_root.iterdir(), reverse=True):
            if child.is_dir():
                candidates.append(child)

    for candidate in candidates:
        if (candidate / "verification" / "candidates").exists() or (
            candidate / "selection" / "winner" / "rules.json"
        ).exists():
            return candidate
    return None


def _verification_candidate_rules(
    pipeline_dir: Optional[Path],
) -> List[Optional[Dict[str, Any]]]:
    if pipeline_dir is None:
        return []

    candidates_dir = pipeline_dir / "verification" / "candidates"
    if not candidates_dir.exists():
        return []

    rules = []  # type: List[Optional[Dict[str, Any]]]
    for candidate_dir in sorted(candidates_dir.glob("candidate_*")):
        if candidate_dir.is_dir():
            rules.append(_load_json_file(candidate_dir / "rules.json"))
    return rules


def _winner_rules(
    row: Dict[str, Any],
    pipeline_dir: Optional[Path],
) -> Optional[Dict[str, Any]]:
    from_summary = _load_json_text(row.get("winner_rules"))
    if from_summary is not None:
        return from_summary

    if pipeline_dir is None:
        return None

    return _load_json_file(pipeline_dir / "selection" / "winner" / "rules.json")


def _same_candidate_ratio(
    candidate_rules: List[Optional[Dict[str, Any]]],
) -> Tuple[str, int, int]:
    canonical = [_canonical_rules(rules) for rules in candidate_rules]
    valid = [item for item in canonical if item is not None]
    total = len(canonical)

    if total == 0:
        return "", 0, 0
    if not valid:
        return f"0/{total}", 0, total

    largest_group = Counter(valid).most_common(1)[0][1]
    return f"{largest_group}/{total}", largest_group, total


def audit(results_dir: Path) -> List[Dict[str, Any]]:
    summary_path = results_dir / "summary.json"
    rows = json.loads(summary_path.read_text())
    if not isinstance(rows, list):
        raise ValueError(f"{summary_path} must contain a JSON list")

    output_rows = []  # type: List[Dict[str, Any]]
    for row in rows:
        row_id = str(row.get("row_id", ""))
        pipeline_dir = _find_pipeline_dir(results_dir, row_id, row.get("pipeline_dir"))

        correct = _load_json_text(row.get("correct_spec"))
        winner = _winner_rules(row, pipeline_dir)
        canonical_winner = _canonical_rules(winner)
        canonical_correct = _canonical_rules(correct)
        winner_correct = (
            canonical_winner is not None
            and canonical_correct is not None
            and canonical_winner == canonical_correct
        )

        verification_rules = _verification_candidate_rules(pipeline_dir)
        same_ratio, same_count, candidate_count = _same_candidate_ratio(verification_rules)

        output_rows.append({
            "row_id": row_id,
            "winner_correct": winner_correct,
            "verification_same_candidates": same_ratio,
            "verification_same_count": same_count,
            "verification_candidate_count": candidate_count,
            "pipeline_dir": str(pipeline_dir) if pipeline_dir else "",
        })

    return output_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit winner correctness and verification-candidate rule agreement."
    )
    parser.add_argument("results_dir", type=Path, help="Experiment results folder")
    parser.add_argument(
        "--output",
        type=Path,
        help="CSV output path; defaults to <results_dir>/verification_audit.csv",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    output_path = args.output or (results_dir / "verification_audit.csv")

    rows = audit(results_dir)
    fieldnames = [
        "row_id",
        "winner_correct",
        "verification_same_candidates",
        "verification_same_count",
        "verification_candidate_count",
        "pipeline_dir",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
