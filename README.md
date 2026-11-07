# Truth Trails frontend

Static HTML/CSS/JavaScript interface for the local FastAPI service.

## Run

Start the backend from `../backend`:

```sh
source .venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Then, from this folder, serve the frontend:

```sh
python3 -m http.server 5173 --bind 127.0.0.1
```

Open <http://127.0.0.1:5173>. The API address is set once in the
`api-base-url` meta tag in `index.html`.

## Features

- Image upload with preview, type checks, and a 10 MB limit.
- Experimental video upload with preview, a 100 MB client-side limit, and a
  30-second server-side limit.
- Structured image/video report, model/version details, score explanation,
  file SHA-256, and optional user-supplied source reference.
- Browser-local history for up to 20 report records; uploaded media bytes are
  not stored in history.
- Print report, research snapshot, methodology, and limitations.
- Six-model comparison table from the internal presentation, clearly marked as
  reported results that have not been independently rerun here.

Video scores are exploratory: the still-image classifier has not been
validated on video frames, the aggregation method is unvalidated, and audio is
not analyzed. Source references are not verified. Blockchain, watermark
detection, and community review are not connected.
