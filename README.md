# Lumenbrain

An AI-assisted pipeline for glioma MRI: upload four co-registered brain MRI
volumes, get a 3D tumour segmentation, and get that segmentation explained
against a citation-backed evidence corpus — not against a language model's
own memory.

Runs entirely on your own machine. Research and educational use only. Not a
medical device, not for diagnosis.

## Core problem this solves

Two separate problems, addressed by two separate mechanisms rather than one
model asked to do both:

1. **Where is the tumour, and how big is it?** A trained segmentation model
   answers this from the MRI itself. This is the kind of question a single
   scan can actually answer, and the model's own reliability at it is
   measured (holdout Dice per region — see [Model provenance](#model-provenance)),
   not assumed.

2. **What does this finding mean, clinically?** A language model answering
   from its own weights can sound confident and still be wrong, and there is
   no way for a clinician to check which. Lumenbrain's evidence agent
   never answers a clinical question from memory — every claim either cites
   a specific passage from NCI PDQ / PubMed / PMC / a domain-restricted web
   search, or is explicitly listed as not addressed by the current evidence.
   Two questions in particular — **how long has this tumour been growing**
   and **what is the survival outlook** — are not retrieval gaps to paper
   over with a plausible-sounding citation. No single-timepoint MRI can
   answer either one, so the app says exactly that, deterministically, every
   time, instead of generating an answer that merely looks grounded.

The thing being solved for is trustworthiness under uncertainty, not just
"segment a tumour" or "chat about cancer" — both of those already exist
elsewhere. See [Honest limits](#honest-limits) for what this project does
*not* claim to do.

## Architecture

```
browser
  │  four .nii.gz files
  ▼
Next.js  (localhost:3001)
  │  POST /api/predict
  ▼
FastAPI  (localhost:8000)
  │
  ├─ inference.py    RAS → 1 mm → crop → normalise → DynUNet, TTA, component gate
  ├─ meshing.py       label map → marching cubes → brain.glb + tumor.glb
  │                   + per-region confidence (see below)
  ▼
brain.glb · tumor.glb · mask.nii.gz · metadata JSON
  │
  ▼
three.js viewer  ──┐
                    │  POST /api/cases/{id}/explain   (5 fixed questions)
                    │  POST /api/cases/{id}/ask        (follow-up, free text)
                    ▼
              knowledge_routes.py
                    │
        ┌───────────┼────────────────────────┐
        ▼           ▼                        ▼
  clinical_agent  planner → hybrid_search  web_search
  (Phase 4:       (BM25 + dense, RRF)      (Tavily, domain-
   no retrieval,        │                   restricted allowlist)
   hardcoded refusal    ▼
   on growth/         answer_question → Evidence Validator
   survival)          (citation-gated, numbered citations
                        resolved server-side, claims flagged
                        not silently dropped)
```

## Technology used

**Segmentation (backend, Phase 1)**
- [MONAI](https://monai.io/) `DynUNet` (nnU-Net-style 3D CNN), PyTorch,
  trained on BraTS 2024 GLI+MEN+PED combined (adult glioma, meningioma,
  pediatric glioma), plus a post-hoc learned component gate
- `nibabel` / `scipy.ndimage` / `scikit-image` (marching cubes) / `trimesh`
  for MRI → 3D mesh conversion
- FastAPI serving `.glb` meshes and JSON metadata

**Evidence retrieval (backend, Phase 3)**
- Hybrid retrieval: SQLite FTS5 (BM25) + dense embeddings
  (`sentence-transformers`, MiniLM — empirically selected over MedCPT on this
  corpus) fused by reciprocal rank fusion, via [ChromaDB](https://www.trychroma.com/)
- Ingestion: NCI PDQ (manual fixture-only — NCBI Bookshelf forbids automated
  crawling), PubMed/PMC via NCBI E-utilities + BioC API (both explicitly
  permit automated retrieval), live web search via
  [Tavily](https://tavily.com) restricted server-side to an authoritative
  medical-domain allowlist (cancer.gov, nih.gov, who.int, Cochrane, ASCO,
  ASTRO, etc.) — never open web search
- LLM answer generation: Groq (`openai/gpt-oss-120b`, free tier, primary) with
  automatic fallback to Anthropic (`claude-sonnet-5`) if Groq is unavailable
  or rate-limited; a deterministic `StubClient` for offline pipeline testing

**Clinical analysis agent (backend, Phase 4)**
- No retrieval — a single constrained LLM call for three sections (what the
  finding generally means, how the location generally relates to brain
  function, typical treatment categories), plus two sections that are
  hardcoded and never touch the model at all (growth duration, survival)

**Frontend**
- Next.js 14 (App Router), React 18
- React Three Fiber + drei + three.js for the interactive 3D viewer,
  `@react-three/postprocessing` for bloom/glow effects
- Inline JS style objects — no Tailwind/CSS framework

## Setup

### 1. Backend — imaging pipeline

```bash
cd backend
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Copy `best_lesion.pt` (132 MB) and `component_gate.joblib` from your Kaggle
download into `backend/models/`. If a filename downloaded with a stray space
or parentheses (e.g. `best_lesion (1).pt`), rename it — those break path
handling.

```bash
uvicorn app.main:app --reload --port 8000
```

Check it: <http://localhost:8000/api/health>

```json
{ "status": "ok", "checkpoint": "best_lesion.pt", "checkpoint_present": true, "component_gate_active": true }
```

The checkpoint loads lazily on the first prediction, so `device` reads
`not loaded yet` until then. That is expected.

### 2. Backend — evidence pipeline (optional but needed for `/explain` and `/ask`)

```bash
cd backend
pip install -r requirements-knowledge.txt
cp .env.example .env        # fill in at least one LLM key — see below
python -m app.knowledge.cli build-corpus   # ingest -> chunk -> chunks.jsonl
```

Then build the retrieval indexes (BM25 + dense) — see `app/knowledge/index/`.
Without this step, the app still runs: segmentation and the 3D viewer work,
`/explain`'s hardcoded growth/survival sections still work, but the
retrieval-backed sections degrade gracefully to "answer service unavailable"
rather than failing the request.

Environment variables (`backend/.env`, see `.env.example` for the full
annotated version):

| variable | required? | purpose |
| --- | --- | --- |
| `GROQ_API_KEY` | one of these two | primary LLM, free tier |
| `ANTHROPIC_API_KEY` | one of these two | fallback LLM if Groq is down/rate-limited |
| `NCBI_API_KEY` | optional | raises PubMed/PMC E-utilities rate limit 3→10 req/s |
| `TAVILY_API_KEY` | optional | enables live web search; omitted = curated corpus only |

### 3. Frontend

```bash
cd frontend
npm install
cp .env.local.example .env.local
npm run dev -- -p 3001
```

Open <http://localhost:3001>. **Must be port 3001** — the backend's CORS
allowlist (`app/main.py`, `ALLOW_ORIGINS`) only accepts that origin by
default.

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

## Segmentation confidence (per-region, per-case)

Aggregate holdout Dice tells you how the model performs on average — it does
not tell you whether *this specific scan's* resection-cavity boundary is one
of the trustworthy ones. Two things measure and surface that gap:

- **`inference.py`** runs test-time-augmentation passes (identity + one flip
  per spatial axis, up to 4) and measures how much the model disagrees with
  itself under a symmetry it should be invariant to, separately per region
  (TC/WT/ET/RC). Each pass is a full extra forward pass over the volume — the
  dominant cost of a request on CPU — so this is a real tradeoff, controlled
  by `TTA_PASSES` (default `2`, i.e. roughly 2x a single pass instead of 4x).
  Set `TTA_PASSES=1` to disable TTA entirely and get the original single-pass
  speed back; region uncertainty then reports `None` rather than a
  zero-from-one-sample that would misreport "measured, no disagreement."
- **`clinical_agent.py`** reads that per-region score and, only for regions
  above a fixed threshold, appends a specific hedge to the `/explain`
  response's warnings (e.g. "ET boundary showed elevated disagreement ...
  treat this region's extent as less reliable than the others in this
  case") — instead of a blanket disclaimer that applies to every case
  identically.

This score is explicitly **not** a calibrated error probability — no
per-case holdout data exists yet to fit one (would need per-case, per-region
ground-truth Dice on the 150-case holdout set, which the current eval script
does not save). It is a relative, within-case disagreement signal, labelled
as such everywhere it's surfaced (`meshing.py`'s `meta["confidence"]`,
`clinical_agent.py`'s `segmentation_confidence`).

## Evidence agent — how a question gets answered

`POST /api/cases/{id}/explain` runs five fixed questions automatically:

1. What is this tumour? *(LLM, general medical education, no citations)*
2. How might it affect this part of the brain? *(LLM, same)*
3. How long has it been growing? *(hardcoded — no MRI answers this)*
4. What's the survival outlook? *(hardcoded — no MRI answers this)*
5. What treatments are typically used? *(LLM, general medical education)*

`POST /api/cases/{id}/ask` answers anything else through the Phase 3
pipeline instead: a rule-based planner routes the question (imaging-only,
NCI-only, research-only, or full-corpus), `hybrid_search` retrieves
passages, the LLM drafts an answer under a citation-gated contract enforced
in code (not just prompted for), and an Evidence Validator cross-checks each
claim's numbers/entities against its own cited passage — a claim that fails
gets flagged for review, never silently dropped or silently approved.

## API

```
GET  /api/health
POST /api/predict                          multipart: t1n, t1c, t2w, t2f
GET  /api/jobs/{id}/brain.glb
GET  /api/jobs/{id}/tumor.glb
GET  /api/jobs/{id}/mask.nii.gz
POST /api/cases/{job_id}/explain            5 fixed questions, see above
POST /api/cases/{job_id}/ask                free-text follow-up, {"question": "..."}
```

`/api/predict` returns per-region volumes in cm³, coarse locations, mean
predicted probabilities, per-region TTA uncertainty, and URLs for the three
assets.

## Honest limits

`location` is a threshold heuristic on MNI152 coordinates, not an atlas
segmentation. It supports "left frontal", not a named gyrus. It is omitted
entirely when the input grid is not MNI152 1 mm.

`mean_probability` is the average sigmoid output inside each predicted
region. It is not a calibrated confidence and must never be labelled as one.

`tta_uncertainty` (see above) is a raw self-disagreement score, also not a
calibrated confidence, for the same reason.

`holdout_select_dice` describes the model in general, not this scan — it is
never rephrased as per-case confidence anywhere in the pipeline.

Volumes are measured in the 1 mm resampled space the model works in, not the
original voxel grid.

RC (resection cavity) is the weakest region at Dice 0.6237. It was excluded
from the model-selection metric, so the model was never optimised for it.

Growth rate and survival/prognosis are never estimated by any model in this
pipeline, for any case, under any circumstance — not a conservative default,
a structural fact about what one segmentation of one scan can support.

This is applied engineering with real measured evaluation (holdout Dice,
retrieval Recall@20, citation precision, faithfulness, abstention rate on a
40-question hand-written eval set) — not a peer-reviewed novel research
contribution, and the individual components (DynUNet, hybrid RAG,
biomedical embeddings) are established techniques, not new ones. Prior art
exists for both halves separately (e.g. BraTS Toolkit for segmentation +
3D visualization, guideline-grounded oncology RAG chatbots for the evidence
side); this project's specific combination and its hardcoded refusal on
growth/survival is the part that isn't off-the-shelf.

## Before deploying anywhere public

This is a development server. It has no authentication, no rate limiting, and
it never deletes finished jobs. Uploaded scans and derived masks stay on disk
in `backend/jobs/` indefinitely. Patient imaging carries real legal
obligations — add auth, a retention policy, and TLS before this touches a
network you do not control.

The Groq/Anthropic API keys in `backend/.env` are real credentials with
billing/rate-limit consequences attached — `.env` is gitignored, keep it
that way, and never commit it.

## Model provenance

DynUNet (MONAI, nnU-Net-style encoder-decoder; see
`backend/models/traning-notebook.ipynb`), trained on three BraTS 2024 cohorts
combined — BraTS-GLI (adult glioma), BraTS-MEN (meningioma) and BraTS-PED
(pediatric glioma) — with no tumor-type classification head. Best checkpoint
selected on mean Dice over TC/WT/ET at epoch 96 (of 110); a separate
best-lesion-wise checkpoint was tracked independently (epoch 111).

The raw model's voxel-wise output is substantially worse pooled across three
cohorts than the old single-cohort model was on glioma alone, so a learned
per-lesion "component gate" (`HistGradientBoostingClassifier`,
`backend/models/component_gate.joblib`) filters predicted connected
components post-hoc before meshing — see `backend/app/inference.py`'s module
docstring for why this gate is mandatory, not optional, overhead.

Evaluated on 293 held-out cases, disjoint from training at the patient level
(`backend/models/test_lesionwise.csv`), lesion-wise Dice after gating:

| region | Dice (gated) | Dice (ungated) |
| --- | --- | --- |
| WT | 0.7504 | 0.5062 |
| TC | 0.7045 | 0.5548 |
| ET | 0.6768 | 0.5282 |
| RC | 0.4584 | 0.2622 |

Selection Dice (mean of TC/WT/ET, gated, pooled across cohorts): 0.7106.

Per-cohort pooled Dice (gated) — note the spread; this is three different
tasks pooled into one number above, not one task evaluated three times:

| cohort | Dice |
| --- | --- |
| BraTS-MEN (meningioma) | 0.8065 |
| BraTS-GLI (adult glioma) | 0.6354 |
| BraTS-PED (pediatric glioma) | 0.6292 |

RC (resection cavity) was only present/validated against ground truth in the
BraTS-GLI cohort — treat RC predictions on meningioma or pediatric-glioma
scans with extra caution.
