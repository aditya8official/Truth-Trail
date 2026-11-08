# Larger image dataset training

## Current status

The application already uses OpenCV to decode uploaded images, resize them to
128×128, sample video frames, and detect/crop faces. It did not previously
contain a reproducible training script. The new training script also uses
OpenCV to resize dataset images, then fine-tunes a MobileNetV2 classifier whose
sigmoid output is the application's fake score.

The training workflow uses the public
[Identity-Isolated Deepfake Face Images dataset](https://huggingface.co/datasets/ThinothW/Deepfake-Identity-Isolated-Dataset-PreP),
pinned at revision `72ceb5929d8650d38f91ff5698dda68f96403c51`. Its card reports
142,837 images (71,417 real and 71,420 fake), with face-swap and entire-face
synthesis examples and train/validation/test partitions. The card claims
identity-isolated splits; this project does not independently verify those
identity assignments.

The dataset card states academic/research use only and says the original
source-dataset terms still apply. Keep downloaded images and resulting model
weights local; neither is included in Git. Check each source's terms before
redistribution or other use.

## Reproduce training

Install backend dependencies, including Hugging Face Datasets with image
support, then run:

```sh
cd backend
.venv/bin/python -m pip install -r requirements.txt
cd ..
backend/.venv/bin/python research/train_mobilenetv2_hf.py
```

The script downloads the pinned dataset and ImageNet initialization weights if
they are not cached. It uses the official dataset splits, maps the source's
`0=fake, 1=real` labels to the application's `1=fake, 0=real` score, and uses
OpenCV to resize the 260×260 images to the application's existing 128×128
input size. The published parquet rows are ordered by class, so the loader
interleaves sequential fake and real streams with seeded pair ordering before
the TensorFlow shuffle buffer. It evaluates both the existing model and the
new candidate on the same held-out test split when the current model file is
present.

Defaults are batch size 16, two classifier-head epochs, and three fine-tuning
epochs over the last 30 MobileNetV2 layers. A validation-AUC checkpoint is
saved separately as
`backend/models/mobilenetv2_dfds_candidate.keras`. The training log and JSON
report go under `backend/training_runs/`. The script refuses to overwrite an
existing candidate model or run directory. It does not change the model the
API loads by default.

Downloaded data and resized `tf.data` caches live under `backend/data/` and are
ignored by Git. The current machine has an Apple M3 CPU, 8 GB RAM, no TensorFlow
GPU device, and about 54 GB free disk at audit time. The dataset itself is
reported as about 2.87 GB; cache files and model training add local storage
use, and CPU training can take a while.

## Evaluation limits

The validation split selects the saved checkpoint. The test split is used only
for final comparison. The completed run used 99,910 training, 21,611
validation, and 21,316 test images. The candidate achieved test ROC-AUC 0.752,
compared with 0.553 for the existing model on the same split. At threshold 0.5,
candidate fake recall was 78.4% and real specificity was 55.8%; the existing
model had 15.9% recall and 87.0% specificity. The candidate therefore finds
more fakes but also incorrectly flags many more real images.

`research/calibrate_threshold.py` selects cutoffs using validation data only
and then reports their performance on the held-out test split. A cutoff of
0.620 maximized validation balanced accuracy; on test it gave 67.7% accuracy,
66.8% real specificity, and 68.6% fake recall. To target at least 90%
specificity on validation, the selected cutoff was 0.883; on test it yielded
90.8% specificity and 37.1% fake recall. At a 95% validation specificity
target, test specificity was 95.4% and fake recall was 27.0%.

The sigmoid values are uncalibrated ranking scores, not probabilities, and
these cutoffs are dataset-specific rather than deployment settings. The new
weights remain a local research candidate and the API default has not changed.
These measurements describe this dataset's held-out split; they do not prove
performance on arbitrary social-media images or video. The video endpoint
continues to use the still-image model and remains unvalidated. The run and
threshold reports are stored locally under `backend/training_runs/` and are
excluded from Git along with the downloaded data and model weights.
