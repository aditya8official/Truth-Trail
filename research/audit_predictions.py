#!/usr/bin/env python3
"""Recompute metrics from saved MobileNetV2 predictions without training or downloads."""

import argparse
import csv
import json
from pathlib import Path


PREDICTION_FILES = (
    "mobilenetv2_test_predictions.csv",
    "mobilenetv2_finetuned_test_predictions.csv",
)
PREDICTION_COLUMNS = {"path", "true_label", "fake_probability", "predicted_label"}
FAKE_FILENAME_GROUPS = {"easy", "mid", "hard"}


def _auc(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError("ROC-AUC requires examples from both classes.")

    ordered = sorted(zip(scores, labels), key=lambda pair: pair[0])
    positive_rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2
        positive_rank_sum += average_rank * sum(label for _, label in ordered[index:end])
        index = end

    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def _average_precision(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    if not positives:
        raise ValueError("Average precision requires at least one positive example.")

    ordered = sorted(zip(scores, labels), key=lambda pair: pair[0], reverse=True)
    true_positives = 0
    seen = 0
    average_precision = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        group_positives = sum(label for _, label in ordered[index:end])
        true_positives += group_positives
        seen += end - index
        average_precision += (group_positives / positives) * (true_positives / seen)
        index = end

    return average_precision


def _compute_metrics(labels: list[int], scores: list[float], predictions: list[int]) -> dict:
    true_negatives = sum(label == 0 and prediction == 0 for label, prediction in zip(labels, predictions))
    false_positives = sum(label == 0 and prediction == 1 for label, prediction in zip(labels, predictions))
    false_negatives = sum(label == 1 and prediction == 0 for label, prediction in zip(labels, predictions))
    true_positives = sum(label == 1 and prediction == 1 for label, prediction in zip(labels, predictions))

    total = len(labels)
    positives = true_positives + false_negatives
    negatives = true_negatives + false_positives
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
    recall = true_positives / positives if positives else 0.0
    real_recall = true_negatives / negatives if negatives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "accuracy": (true_positives + true_negatives) / total,
        "balanced_accuracy": (real_recall + recall) / 2,
        "fake_precision": precision,
        "fake_recall": recall,
        "fake_f1": f1,
        "roc_auc": _auc(labels, scores),
        "average_precision": _average_precision(labels, scores),
        "confusion_matrix_labels_0_1": [
            [true_negatives, false_positives],
            [false_negatives, true_positives],
        ],
    }


def _load_manifest(manifest_path: Path) -> list[dict]:
    manifest = json.loads(manifest_path.read_text())
    test_rows = manifest.get("test") if isinstance(manifest, dict) else None
    if not isinstance(test_rows, list):
        raise ValueError("The split manifest must contain a test row list.")
    return test_rows


def _audit_prediction_file(path: Path, test_rows: list[dict], threshold: float) -> dict:
    with path.open(newline="") as prediction_file:
        reader = csv.DictReader(prediction_file)
        if not PREDICTION_COLUMNS.issubset(reader.fieldnames or []):
            raise ValueError(f"{path.name} is missing required prediction columns.")
        rows = list(reader)

    labels = [int(row["true_label"]) for row in rows]
    scores = [float(row["fake_probability"]) for row in rows]
    predictions = [int(row["predicted_label"]) for row in rows]
    if any(label not in (0, 1) for label in labels + predictions):
        raise ValueError(f"{path.name} contains labels outside the binary 0/1 mapping.")
    if any(not 0 <= score <= 1 for score in scores):
        raise ValueError(f"{path.name} contains scores outside [0, 1].")

    path_and_label_match = len(rows) == len(test_rows) and all(
        row["path"] == manifest_row["path"]
        and labels[index] == int(manifest_row["label"])
        for index, (row, manifest_row) in enumerate(zip(rows, test_rows))
    )
    threshold_labels_match = all(
        prediction == int(score >= threshold)
        for prediction, score in zip(predictions, scores)
    )
    groups: dict[str, dict[str, int]] = {}
    for row, label, prediction in zip(rows, labels, predictions):
        if label == 0:
            group_name = "real"
        else:
            filename_group = Path(row["path"]).stem.split("_", 1)[0].lower()
            group_name = filename_group if filename_group in FAKE_FILENAME_GROUPS else "fake_other"
        counts = groups.setdefault(
            group_name,
            {"count": 0, "correct": 0, "false_positives": 0, "false_negatives": 0},
        )
        counts["count"] += 1
        counts["correct"] += int(label == prediction)
        counts["false_positives"] += int(label == 0 and prediction == 1)
        counts["false_negatives"] += int(label == 1 and prediction == 0)

    group_order = {"real": 0, "easy": 1, "mid": 2, "hard": 3, "fake_other": 4}
    group_breakdown = [
        {
            "filename_group": name,
            **counts,
            "class_specific_accuracy": counts["correct"] / counts["count"],
        }
        for name, counts in sorted(groups.items(), key=lambda item: group_order.get(item[0], 99))
    ]

    return {
        "file": path.name,
        "rows": len(rows),
        "path_order_and_true_labels_match_test_manifest": path_and_label_match,
        "predicted_labels_match_threshold": threshold_labels_match,
        "threshold": threshold,
        "metrics": _compute_metrics(labels, scores, predictions),
        "filename_group_breakdown": group_breakdown,
    }


def _read_traditional_summary(path: Path) -> list[dict]:
    with path.open(newline="") as summary_file:
        reader = csv.DictReader(summary_file)
        required = {
            "Model", "Accuracy", "Balanced Accuracy", "Fake Precision",
            "Fake Recall", "Fake F1", "ROC-AUC", "Average Precision",
        }
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path.name} is missing expected aggregate metric columns.")
        return [row for row in reader]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="Saved JSON split manifest.")
    parser.add_argument("--results-dir", required=True, type=Path, help="Folder containing prediction CSVs.")
    parser.add_argument("--traditional-metrics", type=Path, help="Optional saved aggregate metrics CSV for traditional models.")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    test_rows = _load_manifest(args.manifest)
    prediction_audits = [
        _audit_prediction_file(args.results_dir / filename, test_rows, args.threshold)
        for filename in PREDICTION_FILES
    ]
    output = {
        "manifest_test_rows": len(test_rows),
        "prediction_audits": prediction_audits,
    }
    if args.traditional_metrics:
        output["traditional_models_saved_aggregate_metrics"] = _read_traditional_summary(args.traditional_metrics)

    print(json.dumps(output, indent=2))
    consistent = all(
        audit["path_order_and_true_labels_match_test_manifest"]
        and audit["predicted_labels_match_threshold"]
        for audit in prediction_audits
    )
    return 0 if consistent else 1


if __name__ == "__main__":
    raise SystemExit(main())
