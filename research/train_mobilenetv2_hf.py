"""Fine-tune the Truth Trails MobileNetV2 on a larger HF face dataset.

Dataset images are decoded and resized with OpenCV. The dataset's published
label convention is 0=fake, 1=real; the application model emits a fake score,
so labels are inverted during loading. Data, caches, checkpoints, and reports
stay outside Git.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import random
import sys
from typing import Iterator
from itertools import zip_longest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
DEFAULT_CACHE_DIR = BACKEND_ROOT / "data" / "hf-cache"
DEFAULT_MODEL_OUTPUT = BACKEND_ROOT / "models" / "mobilenetv2_dfds_candidate.keras"
DEFAULT_BASELINE = BACKEND_ROOT / "models" / "mobilenetv2_finetuned_best.keras"
DATASET_ID = "ThinothW/Deepfake-Identity-Isolated-Dataset-PreP"
DATASET_REVISION = "72ceb5929d8650d38f91ff5698dda68f96403c51"
IMAGE_SIDE = 128

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ.setdefault("HF_HOME", str(DEFAULT_CACHE_DIR))

import cv2
import numpy as np
import tensorflow as tf
from datasets import load_dataset


LOGGER = logging.getLogger("truthtrails.training")


def _fake_label(raw_label: int | str, label_names: list[str] | None) -> np.int32:
    if isinstance(raw_label, str):
        normalized = raw_label.strip().lower()
        if normalized not in {"fake", "real"}:
            raise ValueError(f"Unexpected class label: {raw_label!r}")
        return np.int32(normalized == "fake")

    label = int(raw_label)
    if label_names:
        if label < 0 or label >= len(label_names):
            raise ValueError(f"Class label index is out of range: {label}")
        return np.int32(label_names[label].strip().lower() == "fake")

    # Dataset card convention when the Arrow feature omits class names.
    if label not in (0, 1):
        raise ValueError(f"Unexpected binary class label: {label}")
    return np.int32(label == 0)


def _dataset_label_names(dataset_split) -> list[str] | None:
    feature = dataset_split.features.get("label")
    names = getattr(feature, "names", None)
    if names is None:
        return None
    normalized = [name.strip().lower() for name in names]
    if set(normalized) != {"fake", "real"}:
        raise ValueError(f"Unexpected dataset label names: {names!r}")
    return normalized


def _balanced_training_rows(split, label_names: list[str] | None, seed: int):
    """Read the class-ordered Arrow table sequentially while interleaving labels."""
    source_labels = np.asarray(split["label"], dtype=np.int32)
    fake_index = label_names.index("fake") if label_names else 0
    real_index = label_names.index("real") if label_names else 1
    fake_rows = split.select(np.flatnonzero(source_labels == fake_index).tolist())
    real_rows = split.select(np.flatnonzero(source_labels == real_index).tolist())
    rng = random.Random(seed)

    for fake_row, real_row in zip_longest(fake_rows, real_rows):
        pair = [fake_row, real_row]
        if rng.getrandbits(1):
            pair.reverse()
        for row in pair:
            if row is not None:
                yield row


def _make_data_pipeline(split, split_name: str, label_names: list[str] | None,
                        cache_dir: Path, batch_size: int, seed: int):
    output_signature = (
        tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )

    def examples() -> Iterator[tuple[np.ndarray, np.int32]]:
        source = (
            _balanced_training_rows(split, label_names, seed)
            if split_name == "train"
            else iter(split)
        )
        for row in source:
            pil_image = row["image"].convert("RGB")
            image = np.asarray(pil_image, dtype=np.uint8).copy()
            yield image, _fake_label(row["label"], label_names)

    dataset = tf.data.Dataset.from_generator(examples, output_signature=output_signature)

    def opencv_resize(image, label):
        def resize(array):
            if hasattr(array, "numpy"):
                array = array.numpy()
            array = np.asarray(array, dtype=np.uint8)
            resized = cv2.resize(
                array,
                (IMAGE_SIDE, IMAGE_SIDE),
                interpolation=cv2.INTER_AREA,
            )
            return np.ascontiguousarray(resized, dtype=np.uint8)

        resized = tf.py_function(resize, [image], Tout=tf.uint8)
        resized.set_shape((IMAGE_SIDE, IMAGE_SIDE, 3))
        return resized, label

    options = tf.data.Options()
    options.deterministic = True
    options.threading.private_threadpool_size = 4
    dataset = dataset.with_options(options)
    dataset = dataset.map(opencv_resize, num_parallel_calls=4)
    cache_name = (
        f"train_balanced_seed{seed}_{IMAGE_SIDE}px.cache"
        if split_name == "train"
        else f"{split_name}_{IMAGE_SIDE}px.cache"
    )
    dataset = dataset.cache(str(cache_dir / cache_name))

    if split_name == "train":
        dataset = dataset.shuffle(2048, seed=seed, reshuffle_each_iteration=True)
    if split_name in {"train", "validation"}:
        dataset = dataset.repeat()

    def mobilenet_preprocess(images, labels):
        images = tf.cast(images, tf.float32)
        images = tf.keras.applications.mobilenet_v2.preprocess_input(images)
        return images, tf.cast(labels, tf.float32)

    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.map(mobilenet_preprocess, num_parallel_calls=2)
    return dataset.prefetch(1)


def _build_model() -> tuple[tf.keras.Model, tf.keras.Model]:
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(IMAGE_SIDE, IMAGE_SIDE, 3),
        include_top=False,
        weights="imagenet",
    )
    backbone.trainable = False

    inputs = tf.keras.Input(shape=(IMAGE_SIDE, IMAGE_SIDE, 3), name="rgb_image")
    features = backbone(inputs, training=False)
    features = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pool")(features)
    features = tf.keras.layers.Dropout(0.25, name="head_dropout")(features)
    features = tf.keras.layers.Dense(64, activation="relu", name="dense_64")(features)
    features = tf.keras.layers.Dropout(0.2, name="dense_dropout")(features)
    fake_score = tf.keras.layers.Dense(1, activation="sigmoid", name="fake_score")(features)
    model = tf.keras.Model(inputs=inputs, outputs=fake_score, name="truth_trails_mobilenetv2")
    return model, backbone


def _compile(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc", num_thresholds=1000),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )


class BestByValidationAUC(tf.keras.callbacks.Callback):
    def __init__(self, destination: Path):
        super().__init__()
        self.destination = destination
        self.best = 0.5
        self.saved_epoch: int | None = None
        self.epoch_count = 0

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_count += 1
        logs = logs or {}
        score = logs.get("val_auc")
        if score is not None and float(score) > self.best:
            self.best = float(score)
            self.saved_epoch = self.epoch_count
            self.model.save(self.destination)
            LOGGER.info("Saved best checkpoint: val_auc=%.6f", self.best)


def _metrics(model: tf.keras.Model, dataset) -> dict:
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    for images, batch_labels in dataset:
        batch_scores = model(images, training=False)
        labels.append(np.asarray(batch_labels).reshape(-1).astype(np.int32))
        scores.append(np.asarray(batch_scores).reshape(-1).astype(np.float64))

    y_true = np.concatenate(labels)
    y_score = np.concatenate(scores)
    y_pred = (y_score >= 0.5).astype(np.int32)
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    true_positive = int(np.sum((y_true == 1) & (y_pred == 1)))
    true_negative = int(np.sum((y_true == 0) & (y_pred == 0)))
    false_positive = int(np.sum((y_true == 0) & (y_pred == 1)))
    false_negative = int(np.sum((y_true == 1) & (y_pred == 0)))

    sorted_indices = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[sorted_indices]
    sorted_labels = y_true[sorted_indices]
    _, starts, group_sizes = np.unique(
        sorted_scores,
        return_index=True,
        return_counts=True,
    )
    ranks = np.empty(len(y_true), dtype=np.float64)
    for start, size in zip(starts, group_sizes):
        average_rank = ((start + 1) + (start + size)) / 2.0
        ranks[sorted_indices[start:start + size]] = average_rank
    auc = (
        (np.sum(ranks[y_true == 1]) - positives * (positives + 1) / 2.0)
        / max(positives * negatives, 1)
    )

    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    specificity = true_negative / max(true_negative + false_positive, 1)
    return {
        "rows": int(len(y_true)),
        "accuracy": float(np.mean(y_pred == y_true)),
        "roc_auc": float(auc),
        "precision_fake": float(precision),
        "recall_fake": float(recall),
        "specificity_real": float(specificity),
        "f1_fake": float(2 * precision * recall / max(precision + recall, 1e-12)),
        "threshold": 0.5,
        "confusion_matrix": {
            "true_real_pred_real": true_negative,
            "true_real_pred_fake": false_positive,
            "true_fake_pred_real": false_negative,
            "true_fake_pred_fake": true_positive,
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-model", type=Path, default=DEFAULT_MODEL_OUTPUT)
    parser.add_argument("--baseline-model", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--head-epochs", type=int, default=2)
    parser.add_argument("--fine-tune-epochs", type=int, default=3)
    parser.add_argument("--fine-tune-layers", type=int, default=30)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-5)
    args = parser.parse_args()

    if args.batch_size < 1 or args.head_epochs < 0 or args.fine_tune_epochs < 0:
        parser.error("Batch size must be positive and epoch counts cannot be negative.")
    if args.head_epochs + args.fine_tune_epochs == 0:
        parser.error("At least one training phase must run.")
    if args.fine_tune_epochs and args.fine_tune_layers < 1:
        parser.error("Fine-tuning requires --fine-tune-layers to be at least 1.")
    if args.output_model.exists():
        parser.error(f"Refusing to overwrite existing model: {args.output_model}")

    cache_dir = args.cache_dir.expanduser().resolve()
    output_model = args.output_model.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_model.parent.mkdir(parents=True, exist_ok=True)

    if args.run_dir:
        run_dir = args.run_dir.expanduser().resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = BACKEND_ROOT / "training_runs" / f"dfds_142k_{timestamp}"
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"Run directory already exists: {run_dir}")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / "training.log")],
    )
    tf.keras.utils.set_random_seed(args.seed)
    cv2.setNumThreads(1)
    try:
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except RuntimeError:
        pass

    LOGGER.info("Loading %s at revision %s", DATASET_ID, DATASET_REVISION)
    dataset = load_dataset(
        DATASET_ID,
        revision=DATASET_REVISION,
        cache_dir=str(cache_dir),
    )
    required_splits = {"train", "validation", "test"}
    if not required_splits.issubset(dataset.keys()):
        raise ValueError(f"Expected splits {sorted(required_splits)}, got {list(dataset.keys())}")

    label_names = _dataset_label_names(dataset["train"])
    LOGGER.info("Dataset rows: %s", {name: len(dataset[name]) for name in sorted(required_splits)})
    LOGGER.info("Label names (0=fake, 1=real in source card): %s", label_names)
    LOGGER.info("OpenCV %s; TensorFlow %s; devices=%s", cv2.__version__, tf.__version__, tf.config.list_physical_devices())

    data = {
        split_name: _make_data_pipeline(
            dataset[split_name], split_name, label_names, cache_dir, args.batch_size, args.seed
        )
        for split_name in ("train", "validation", "test")
    }
    steps_per_epoch = math.ceil(len(dataset["train"]) / args.batch_size)
    validation_steps = math.ceil(len(dataset["validation"]) / args.batch_size)

    baseline_metrics = None
    if args.baseline_model.is_file():
        LOGGER.info("Evaluating existing Truth Trails model on the new held-out test split")
        baseline_model = tf.keras.models.load_model(args.baseline_model, compile=False)
        baseline_metrics = _metrics(baseline_model, data["test"])
        del baseline_model
        gc.collect()
        LOGGER.info("Existing model metrics: %s", baseline_metrics)

    model, backbone = _build_model()
    checkpoint = BestByValidationAUC(output_model)
    history = []

    if args.head_epochs:
        _compile(model, args.head_learning_rate)
        head_history = model.fit(
            data["train"],
            validation_data=data["validation"],
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            epochs=args.head_epochs,
            callbacks=[
                checkpoint,
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_auc", mode="max", patience=2, restore_best_weights=True
                ),
                tf.keras.callbacks.CSVLogger(run_dir / "training.csv"),
            ],
            verbose=2,
            shuffle=False,
        )
        history.extend(head_history.history.get("auc", []))

    if args.fine_tune_epochs:
        backbone.trainable = True
        fine_tune_layers = min(args.fine_tune_layers, len(backbone.layers))
        for layer in backbone.layers[:-fine_tune_layers]:
            layer.trainable = False
        for layer in backbone.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

        _compile(model, args.fine_tune_learning_rate)
        fine_history = model.fit(
            data["train"],
            validation_data=data["validation"],
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            epochs=args.fine_tune_epochs,
            callbacks=[
                checkpoint,
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_auc", mode="max", patience=2, restore_best_weights=True
                ),
                tf.keras.callbacks.CSVLogger(run_dir / "training.csv", append=bool(args.head_epochs)),
            ],
            verbose=2,
            shuffle=False,
        )
        history.extend(fine_history.history.get("auc", []))

    if not output_model.is_file():
        raise RuntimeError("Training ended without saving a best validation checkpoint.")

    del model, backbone
    gc.collect()
    best_model = tf.keras.models.load_model(output_model, compile=False)
    test_metrics = _metrics(best_model, data["test"])
    del best_model
    gc.collect()

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "id": DATASET_ID,
            "revision": DATASET_REVISION,
            "card_label_mapping": "0=fake, 1=real; training target is 1=fake, 0=real",
            "rows": {name: len(dataset[name]) for name in sorted(required_splits)},
            "reported_license": "Academic/research use only; source dataset terms also apply.",
        },
        "training": {
            "architecture": "MobileNetV2 with ImageNet initialization and a binary fake-score head",
            "row_ordering": "Sequential class streams interleaved real/fake with seeded pair order; tf.data shuffle buffer 2048",
            "input_size": [IMAGE_SIDE, IMAGE_SIDE],
            "batch_size": args.batch_size,
            "head_epochs_requested": args.head_epochs,
            "fine_tune_epochs_requested": args.fine_tune_epochs,
            "fine_tune_layers": args.fine_tune_layers,
            "seed": args.seed,
            "best_validation_auc": checkpoint.best,
            "checkpoint_epoch": checkpoint.saved_epoch,
            "opencv_version": cv2.__version__,
            "tensorflow_version": tf.__version__,
            "devices": [f"{device.device_type}:{device.name}" for device in tf.config.list_physical_devices()],
            "training_auc_history": [float(value) for value in history],
        },
        "model": {
            "path": str(output_model),
            "sha256": _sha256(output_model),
            "bytes": output_model.stat().st_size,
        },
        "test_metrics": test_metrics,
        "baseline_metrics_same_test": baseline_metrics,
        "run_dir": str(run_dir),
        "warning": (
            "The dataset card reports identity-isolated splits, but this audit did not independently "
            "verify source identities or the original source terms. The dataset is a research benchmark, "
            "not evidence of performance on all real-world deepfakes or video."
        ),
    }
    report_path = run_dir / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
