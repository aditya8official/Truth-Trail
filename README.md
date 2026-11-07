# Truth Trails

Experimental face-image analysis project with a static web interface and a
local FastAPI service. Predictions are research signals, not proof of
authenticity.

## Run locally

The trained model is not included in this repository. Place an authorized copy
at `backend/models/mobilenetv2_finetuned_best.keras` before starting the API.

Terminal 1 — install backend dependencies and start the API:

```sh
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Terminal 2 — from the repository root, serve the frontend:

```sh
python3 -m http.server 5173 --bind 127.0.0.1
```

Open <http://127.0.0.1:5173>. API docs are at
<http://127.0.0.1:8000/docs>. The frontend API address is configurable through
the `api-base-url` meta tag in `index.html`.

## Current capabilities

- Image upload, preview, validation, and local model inference.
- Experimental video sampling with face crops using the still-image model.
- Structured reports with model/version, raw score meaning, file hash, and an
  optional user-supplied source reference.
- Browser-local history for up to 20 report records; media bytes are not stored
  in history.
- Printable reports, API health, research metrics, and documented limitations.
- Saved test-set error breakdown for real/easy/mid/hard filename groups; the
  fake difficulty categories are subjective, not source identities.

Video analysis is not validated; audio is not analyzed. Source references are
not verified. Blockchain records, watermark detection, community review, and
external source verification are not connected. The six-model comparison
was cross-checked against saved artifacts. MobileNetV2 metrics were recomputed
from per-image predictions; traditional-model entries match a saved aggregate
CSV. See `docs/research_metrics_audit.md`.

## Research and licensing

The dataset page lists CC BY-NC-SA 4.0. Review the intended use and rights
before redistributing data-derived artifacts. The split audit found filename
groups crossing splits, but those groups do not establish confirmed leakage.
See `docs/dataset_provenance.md`, `docs/research_split_audit.md`, and
`docs/model_card.md`.

No dataset, model weights, virtual environment, or private research outputs are
included in this repository.

## Checks

Run focused backend tests from `backend/`:

```sh
.venv/bin/python -m unittest discover -s tests -v
```

Audit an existing split manifest without downloading data:

```sh
python research/audit_split.py /path/to/dataset_split_seed42.json
```

Recompute metrics from the saved prediction CSVs and check their test rows
against the manifest:

```sh
python research/audit_predictions.py \
  --manifest /path/to/dataset_split_seed42.json \
  --results-dir /path/to/results \
  --traditional-metrics /path/to/results/traditional_models_seed42.csv
```
