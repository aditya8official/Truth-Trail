# Saved metrics audit

## Inputs and scope

This read-only audit compared the saved MobileNetV2 prediction CSVs with the
`test` rows in `dataset_split_seed42.json`. It checked row order, labels, and
the saved 0.5 classification threshold, then recomputed metrics from the
per-image labels, scores, and predictions. No training or dataset download was
performed.

Both prediction files contain 307 rows. Their paths and true labels match the
test manifest in order, and their predicted labels match `fake_probability >=
0.5`.

## Recomputed MobileNetV2 results

Label mapping is `0 = real`, `1 = fake`. Confusion matrices use rows for true
labels and columns for predicted labels, ordered `[real, fake]`.

| Model | Accuracy | Balanced accuracy | Fake precision | Fake recall | Fake F1 | ROC-AUC | Average precision | Confusion matrix |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Frozen MobileNetV2 | 64.50% | 63.85% | 64.71% | 53.47% | 58.56% | 0.6797 | 0.6521 | `[[121, 42], [67, 77]]` |
| Fine-tuned MobileNetV2 | 64.17% | 63.75% | 63.08% | 56.94% | 59.85% | 0.7016 | 0.6759 | `[[115, 48], [62, 82]]` |

These values reproduce the rounded MobileNetV2 figures in the presentation and
the more detailed fine-tuned metrics documented in the model card. They are
still results from one saved image-level split, not a fresh training run or an
independent-dataset evaluation.

## Filename-group error breakdown

The dataset's [source README](https://github.com/minostauros/Real-and-Fake-Face-Detection)
describes `easy`, `mid`, and `hard` as subjective fake categories. For the
following table, class-specific accuracy means specificity for the real group
and fake recall for the three fake filename groups.

| Test filename group | Count | Frozen score | Fine-tuned score | Change |
|---|---:|---:|---:|---:|
| Real specificity | 163 | 74.23% | 70.55% | -3.68 pp |
| Fake `easy` recall | 41 | 48.78% | 53.66% | +4.88 pp |
| Fake `mid` recall | 68 | 61.76% | 63.24% | +1.47 pp |
| Fake `hard` recall | 35 | 42.86% | 48.57% | +5.71 pp |

`pp` means percentage points.

These are descriptive results on the same single test split. The filename
groups are subjective, and this breakdown does not identify people, source
photos, or manipulation lineages.

## Traditional-model results

The SVM, KNN, Random Forest, and Logistic Regression rows in
`traditional_models_seed42.csv` match the presentation's rounded accuracy,
ROC-AUC, and fake-recall figures. That CSV contains aggregate metrics only;
there are no per-image traditional-model predictions in the available local
artifacts, so those four models could not be independently recomputed here.

## Reproduce the audit

The script uses only the Python standard library and reads existing files:

```sh
python research/audit_predictions.py \
  --manifest "/path/to/dataset_split_seed42.json" \
  --results-dir "/path/to/results" \
  --traditional-metrics "/path/to/results/traditional_models_seed42.csv"
```

The script does not load the images, download data, or retrain a model.
