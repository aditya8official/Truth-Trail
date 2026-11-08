"""Choose score thresholds from validation data and report held-out test tradeoffs.

This reports decision-threshold tradeoffs for a saved candidate. It does not
train a model, change the API threshold, or use the test split to choose a
threshold.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf
from datasets import load_dataset

import train_mobilenetv2_hf as training


def collect_predictions(model: tf.keras.Model, dataset) -> tuple[np.ndarray, np.ndarray]:
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    for images, batch_labels in dataset:
        labels.append(np.asarray(batch_labels).reshape(-1).astype(np.int32))
        scores.append(
            np.asarray(model(images, training=False)).reshape(-1).astype(np.float64)
        )
    return np.concatenate(labels), np.concatenate(scores)


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    _, starts, sizes = np.unique(sorted_scores, return_index=True, return_counts=True)
    ranks = np.empty(len(y_true), dtype=np.float64)
    for start, size in zip(starts, sizes):
        ranks[order[start:start + size]] = ((start + 1) + (start + size)) / 2.0
    return float(
        (np.sum(ranks[y_true == 1]) - positives * (positives + 1) / 2.0)
        / max(positives * negatives, 1)
    )


def threshold_table(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, dict[str, float]]:
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    order = np.argsort(-y_score, kind="mergesort")
    sorted_scores = y_score[order]
    sorted_labels = y_true[order]
    cumulative_tp = np.cumsum(sorted_labels == 1)
    cumulative_fp = np.cumsum(sorted_labels == 0)
    end_positions = np.r_[
        np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]),
        len(sorted_scores) - 1,
    ]
    thresholds = sorted_scores[end_positions]
    true_positives = cumulative_tp[end_positions]
    false_positives = cumulative_fp[end_positions]
    recalls = true_positives / max(positives, 1)
    specificities = 1.0 - false_positives / max(negatives, 1)
    balanced_accuracies = (recalls + specificities) / 2.0

    best_positions = np.flatnonzero(
        np.isclose(balanced_accuracies, np.max(balanced_accuracies))
    )
    best_balanced_position = best_positions[
        np.argmax(specificities[best_positions])
    ]
    thresholds_for_specificity: dict[str, float] = {
        "max_balanced_accuracy": float(thresholds[best_balanced_position])
    }

    for target in (0.80, 0.90, 0.95):
        eligible = np.flatnonzero(specificities >= target)
        if eligible.size:
            # The last eligible boundary is the lowest threshold that still
            # satisfies the requested validation specificity.
            thresholds_for_specificity[f"specificity_at_least_{target:.2f}"] = float(
                thresholds[eligible[-1]]
            )
    return float(thresholds[best_balanced_position]), thresholds_for_specificity


def metrics_at(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    y_pred = (y_score >= threshold).astype(np.int32)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return {
        "threshold": float(threshold),
        "rows": int(len(y_true)),
        "accuracy": float((tp + tn) / max(len(y_true), 1)),
        "roc_auc": roc_auc(y_true, y_score),
        "precision_fake": float(precision),
        "recall_fake": float(recall),
        "specificity_real": float(specificity),
        "balanced_accuracy": float((recall + specificity) / 2.0),
        "f1_fake": float(2 * precision * recall / max(precision + recall, 1e-12)),
        "confusion_matrix": {
            "true_real_pred_real": tn,
            "true_real_pred_fake": fp,
            "true_fake_pred_real": fn,
            "true_fake_pred_fake": tp,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=training.DEFAULT_MODEL_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=training.DEFAULT_CACHE_DIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model_path = args.model.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    if not model_path.is_file():
        parser.error(f"Model does not exist: {model_path}")
    if args.batch_size < 1:
        parser.error("Batch size must be positive.")

    dataset = load_dataset(
        training.DATASET_ID,
        revision=training.DATASET_REVISION,
        cache_dir=str(cache_dir),
    )
    label_names = training._dataset_label_names(dataset["train"])
    validation = training._make_data_pipeline(
        dataset["validation"], "validation_calibration", label_names,
        cache_dir, args.batch_size, args.seed,
    )
    test = training._make_data_pipeline(
        dataset["test"], "test_calibration", label_names,
        cache_dir, args.batch_size, args.seed,
    )

    model = tf.keras.models.load_model(model_path, compile=False)
    validation_labels, validation_scores = collect_predictions(model, validation)
    test_labels, test_scores = collect_predictions(model, test)

    best_balanced, selected_thresholds = threshold_table(
        validation_labels, validation_scores
    )
    choices = {"fixed_0.50": 0.5, **selected_thresholds}
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": str(model_path),
        "model_sha256": training._sha256(model_path),
        "dataset": training.DATASET_ID,
        "dataset_revision": training.DATASET_REVISION,
        "threshold_selection_split": "validation",
        "test_split_used_for_selection": False,
        "thresholds": {
            name: {
                "selected_threshold": threshold,
                "validation_metrics": metrics_at(
                    validation_labels, validation_scores, threshold
                ),
                "test_metrics": metrics_at(test_labels, test_scores, threshold),
            }
            for name, threshold in choices.items()
        },
        "warning": (
            "Model scores are uncalibrated ranking scores, not probabilities. "
            "Thresholds describe this dataset and are not deployment settings."
        ),
    }

    if args.output:
        output_path = args.output.expanduser().resolve()
        if output_path.exists():
            parser.error(f"Refusing to overwrite existing report: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = training.BACKEND_ROOT / "training_runs" / f"thresholds_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=False)
        output_path = run_dir / "threshold_analysis.json"

    output_path.write_text(json.dumps(report, indent=2) + "\n")
    report["output"] = str(output_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
