# Truth Trails — Media Intelligence UI

A responsive front-end prototype for the Truth Trails academic project: a proposed deepfake-detection and digital-content verification platform.

## Current status

- Responsive static HTML/CSS/JavaScript interface.
- Image selection, drag-and-drop, preview, file type/size validation.
- Research snapshot based on the documented 307-image test set.
- Explicit demo mode: **the UI does not run a trained model**.
- No live AI inference, blockchain, watermarking, or news verification is connected.

## Run locally

Open `index.html` in a modern browser. No build step or package installation is required.

## Publish with GitHub Pages

1. Create a new GitHub repository, for example `truth-trails`.
2. Upload `index.html`, `README.md`, and `.gitignore` to the repository root.
3. In the repository, open **Settings → Pages**.
4. Under build/deployment, choose **Deploy from a branch**.
5. Select branch `main` and folder `/ (root)`, then save.
6. Wait for GitHub Pages to publish the site; the Pages panel will show the public URL.

## Important before making the repository public

- Do not commit datasets, credentials, API keys, private university information, or model files you do not have permission to distribute.
- The included interface contains research metrics from the project's current documented experiments. Verify these figures against your final report before public release.
- The interface is a prototype and must not be presented as an operational deepfake detector.

## Planned next step: connect actual inference

The model currently lives in a Google Colab workflow. Before connecting a live prediction endpoint, identify and export the exact trained model artifact and record its preprocessing requirements, label mapping, and test results. Then build an API (for example, Flask or FastAPI) that loads that artifact and returns a prediction. Do not show model predictions or confidence scores until the API is actually connected and tested.
