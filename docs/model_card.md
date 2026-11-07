# Model card: MobileNetV2 fine-tuned baseline

## Intended use

This is an experimental classifier for still face images. It is suitable for
local research demonstrations, not for high-stakes decisions or definitive
authenticity judgments. The current image model was not trained for video
frames, audio, watermark detection, or source verification.

## Artifact

- Local path: `backend/models/mobilenetv2_finetuned_best.keras`
- Architecture: MobileNetV2 feature extractor and binary classification head
- Input: RGB image, resized to 128 × 128 × 3
- Output: one sigmoid score interpreted as the fake-class score
- Model identifier: first 12 characters of the artifact SHA-256, returned by
  the API as `model_version` (current artifact: `6272ad148a12`)

## Preprocessing and label mapping

The latest saved training notebook applies BGR-to-RGB conversion, resizes with
OpenCV `INTER_AREA`, converts to `float32`, and calls
`tf.keras.applications.mobilenet_v2.preprocess_input`. The local inference code
uses the same sequence.

The saved split manifest encodes `0 = real` and `1 = fake`. The fine-tuning
evaluation uses a fixed `0.5` threshold. The API reports raw scores and does
not claim that they are calibrated probabilities.

## Evaluation snapshot

The presentation and saved prediction artifacts report 307 test images for the
fine-tuned model:

- Accuracy: 64.17%
- Balanced accuracy: 63.75%
- Fake precision: 63.08%
- Fake recall: 56.94%
- Fake F1: 59.85%
- ROC-AUC: 0.7016
- Average precision: 0.6759

These values are recorded experiment results, not evidence of performance on
new datasets. The test images are not present in this local project, so the
model could not be rerun from images or retrained. Metrics were recomputed from
the saved per-image prediction CSV and checked against the split manifest; see
`docs/research_metrics_audit.md`.

## Data and evaluation limitations

The research manifest describes 2,041 images: 1,081 real and 960 fake, split
into 1,428 training, 306 validation, and 307 test images with seed 42. The
dataset's [source documentation](https://github.com/minostauros/Real-and-Fake-Face-Detection)
describes expert-Photoshop composite face images and subjective easy/mid/hard
fake categories. This is not a video deepfake benchmark.

The split is image-level. Filename inspection found numeric groups that cross
splits, but the numeric token is not documented as a source identity. This is a
possible related-image risk, not confirmed leakage. Kaggle metadata identifies
the dataset license as CC BY-NC-SA 4.0; intended use and any rights related to
the depicted people still need review. See
`docs/research_split_audit.md` and `docs/dataset_provenance.md`.

## Video inference limitation

The experimental video route samples frames, uses an OpenCV Haar cascade to
find faces, and applies the still-image model to face crops. Median aggregation
is descriptive and has not been validated. The response names this limitation,
reports frame-level results, and states that audio is not analyzed.

## Responsible-use statement

Predictions may be wrong. A score is not proof of authenticity, intent, or
misinformation. Confirm claims with independent evidence and human review.
