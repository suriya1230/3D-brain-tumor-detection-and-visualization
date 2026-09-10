# NeuroEvidence

Upload four brain MRI volumes, get a tumour segmentation and an interactive 3D
view. Runs entirely on your own machine — no Kaggle, no cloud inference.

Research and educational use only. Not a medical device, not for diagnosis.

```
browser
  │  four .nii.gz files
  ▼
Next.js  (localhost:3000)
  │  POST /api/predict
  ▼
FastAPI  (localhost:8000)
  │
  ├─ inference.py   RAS → 1 mm → crop → normalise → SegResNet sliding window
  ├─ meshing.py     label map → marching cubes → brain.glb + tumor.glb
  │
  ▼
brain.glb · tumor.glb · mask.nii.gz · metadata JSON
  │
  ▼
three.js viewer
```

## Setup

### 1. Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Copy `best_weights.pt` (75 MB) from your Kaggle download into
`backend/models/`. If it downloaded as `best_weights (1).pt`, rename it —
the space and parentheses break path handling.

```bash
uvicorn app.main:app --reload --port 8000
```

Check it: <http://localhost:8000/api/health>

```json
{ "status": "ok", "checkpoint": "best_weights.pt", "checkpoint_present": true }
```

The checkpoint loads lazily on the first prediction, so `device` reads
`not loaded yet` until then. That is expected.

### 2. Frontend

```bash
cd frontend
npm install
cp .env.local.example .env.local
npm run dev
```

Open <http://localhost:3000>.

### Both at once

```bash
./run-dev.sh
```

## VS Code

Open the `neuroevidence` folder as the workspace root, then select the Python
interpreter at `backend/.venv/bin/python` (Ctrl+Shift+P → "Python: Select
Interpreter"). Without that, imports in `app/` show as unresolved.

Recommended extensions: Python, Pylance, ESLint.

## What you upload

Four co-registered volumes from **one** patient and **one** session:

| slot | scan |
| --- | --- |
| `t1n` | T1, no contrast |
| `t1c` | T1 with gadolinium |
| `t2w` | T2 |
| `t2f` | T2-FLAIR |

Roughly 1 mm isotropic, 130+ slices, skull-stripped. BraTS-style filenames
(`case-t1c.nii.gz`) sort themselves into the right slots automatically.

A single scan will not work — the model has four input channels. A CT scan or
a 5 mm-slice clinical series will not work either. That is a data
requirement, not a bug to route around.

## Preprocessing must not drift

`inference.py` reproduces the notebook's val/test chain in this exact order:

```
load → concat → RAS → 1 mm spacing → crop foreground → normalise
```

The crop comes **before** normalisation. Reorder it and you have a different
pipeline, and the reported metrics no longer describe what the server does.

Two more conventions, both in `meshing.py`:

**Axis map `(x, y, z) → (x, z, −y)`.** RAS has z up, three.js has y up. The
permutation preserves handedness, so nothing renders mirrored. A mirrored
brain puts the tumour in the wrong hemisphere.

**Both meshes centre on the brain centroid**, never their own. That is what
keeps the tumour in its real position inside the brain.

## Speed

Sliding-window inference over a whole volume at `overlap=0.5`:

| hardware | time |
| --- | --- |
| CUDA GPU | 3–8 s |
| CPU | 2–5 min |

To trade a little accuracy for speed on CPU, set `SW_OVERLAP=0.25`. Anything
below 0.25 starts showing seams at window boundaries.

## API

```
GET  /api/health
POST /api/predict                multipart: t1n, t1c, t2w, t2f
GET  /api/jobs/{id}/brain.glb
GET  /api/jobs/{id}/tumor.glb
GET  /api/jobs/{id}/mask.nii.gz
```

`/api/predict` returns per-region volumes in cm³, coarse locations, mean
predicted probabilities, and URLs for the three assets.

## Honest limits

`location` is a threshold heuristic on MNI152 coordinates, not an atlas
segmentation. It supports "left frontal", not a named gyrus. It is omitted
entirely when the input grid is not MNI152 1 mm.

`mean_probability` is the average sigmoid output inside each predicted region.
It is not a calibrated confidence and must never be labelled as one.

Volumes are measured in the 1 mm resampled space the model works in, not the
original voxel grid.

RC (resection cavity) is the weakest region at Dice 0.689. It was excluded
from the model-selection metric, so the model was never optimised for it.

## Before deploying anywhere public

This is a development server. It has no authentication, no rate limiting, and
it never deletes finished jobs. Uploaded scans and derived masks stay on disk
in `backend/jobs/` indefinitely. Patient imaging carries real legal
obligations — add auth, a retention policy, and TLS before this touches a
network you do not control.

## Model provenance

SegResNet (MONAI), `init_filters=32`, `blocks_down=(1,2,2,4)`. Trained on 549
cases from 256 patients for 200 epochs; best checkpoint at epoch 160, selected
by mean Dice over TC, WT and ET.

Evaluated on 150 cases from patients disjoint from training at both case and
patient level:

| region | Dice | HD95 |
| --- | --- | --- |
| WT | 0.8966 | 6.58 mm |
| TC | 0.7553 | 8.36 mm |
| ET | 0.7534 | 8.08 mm |
| RC | 0.6886 | 13.12 mm |

Selection Dice 0.8018.
