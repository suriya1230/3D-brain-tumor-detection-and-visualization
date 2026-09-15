"""
NeuroEvidence inference API.

    POST /api/predict            four NIfTI files -> metadata + asset URLs
    GET  /api/jobs/{id}/brain.glb
    GET  /api/jobs/{id}/tumor.glb
    GET  /api/jobs/{id}/mask.nii.gz
    GET  /api/health

Run:  uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .inference import MODALITIES, Segmenter
from .knowledge_routes import router as knowledge_router
from .meshing import build_assets, save_mask_nifti

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(name)s  %(message)s",
)
log = logging.getLogger("neuroevidence")

BASE = Path(__file__).resolve().parent.parent
JOBS = Path(os.getenv("JOB_DIR", BASE / "jobs"))
JOBS.mkdir(parents=True, exist_ok=True)

CKPT = os.getenv("CKPT_PATH") or next(
    (str(BASE / "models" / n)
     for n in ("best_weights.pt", "best.pt")
     if (BASE / "models" / n).exists()),
    str(BASE / "models" / "best_weights.pt"),
)

MAX_MB = int(os.getenv("MAX_UPLOAD_MB", "400"))
SW_OVERLAP = float(os.getenv("SW_OVERLAP", "0.5"))
ALLOWED_SUFFIX = (".nii", ".nii.gz")

app = FastAPI(title="NeuroEvidence", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv(
        "ALLOW_ORIGINS", "http://localhost:3001,http://127.0.0.1:3001"
    ).split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.include_router(knowledge_router)

_segmenter: Segmenter | None = None


def segmenter() -> Segmenter:
    """Lazy singleton: the checkpoint loads once, on the first request."""
    global _segmenter
    if _segmenter is None:
        _segmenter = Segmenter(CKPT)
    return _segmenter


@app.get("/api/health")
def health():
    ok = Path(CKPT).exists()
    body = {
        "status": "ok" if ok else "checkpoint missing",
        "checkpoint": Path(CKPT).name,
        "checkpoint_present": ok,
        "device": str(_segmenter.device) if _segmenter else "not loaded yet",
        "sw_overlap": SW_OVERLAP,
    }
    if _segmenter:
        body["epoch"] = _segmenter.epoch
        body["validation_select_dice"] = round(_segmenter.val_dice, 4)
    return body


def _stash(upload: UploadFile, dest_dir: Path, key: str) -> Path:
    name = (upload.filename or "").lower()
    if not name.endswith(ALLOWED_SUFFIX):
        raise HTTPException(
            400, f"{key}: expected .nii or .nii.gz, got {upload.filename!r}"
        )

    dest = dest_dir / f"{key}.nii.gz" if name.endswith(".gz") else \
        dest_dir / f"{key}.nii"

    size = 0
    limit = MAX_MB * 1024 * 1024
    with dest.open("wb") as fh:
        while chunk := upload.file.read(1 << 20):
            size += len(chunk)
            if size > limit:
                raise HTTPException(413, f"{key} exceeds {MAX_MB} MB")
            fh.write(chunk)
    if size == 0:
        raise HTTPException(400, f"{key} is empty")
    return dest


@app.post("/api/predict")
async def predict(
    t1n: UploadFile = File(..., description="T1 without contrast"),
    t1c: UploadFile = File(..., description="T1 with gadolinium contrast"),
    t2w: UploadFile = File(..., description="T2"),
    t2f: UploadFile = File(..., description="T2-FLAIR"),
    case_id: str | None = None,
):
    uploads = {"t1n": t1n, "t1c": t1c, "t2w": t2w, "t2f": t2f}
    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS / job_id
    job_dir.mkdir(parents=True)
    staging = Path(tempfile.mkdtemp(prefix="ne_"))
    t0 = time.time()

    try:
        paths = {k: str(_stash(f, staging, k)) for k, f in uploads.items()}

        name = case_id or Path(t1c.filename or "case").name
        for suffix in (".nii.gz", ".nii"):
            name = name.replace(suffix, "")
        for mod in MODALITIES:
            name = name.replace(f"-{mod}", "").replace(f"_{mod}", "")
        name = name.strip("-_ ") or f"case-{job_id}"

        log.info("job %s: predicting %s", job_id, name)
        pred = segmenter().predict(paths, sw_overlap=SW_OVERLAP)

        brain_glb, tumor_glb, meta = build_assets(pred, name)
        (job_dir / "brain.glb").write_bytes(brain_glb)
        (job_dir / "tumor.glb").write_bytes(tumor_glb)
        save_mask_nifti(pred, job_dir / "mask.nii.gz")

        meta.update({
            "job_id": job_id,
            "elapsed_s": round(time.time() - t0, 1),
            "assets": {
                "brain": f"/api/jobs/{job_id}/brain.glb",
                "tumor": f"/api/jobs/{job_id}/tumor.glb",
                "mask": f"/api/jobs/{job_id}/mask.nii.gz",
            },
        })
        (job_dir / "case.json").write_text(__import__("json").dumps(meta, indent=2))

        log.info("job %s: done in %.1fs", job_id, meta["elapsed_s"])
        return meta

    except HTTPException:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    except (ValueError, FileNotFoundError) as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        log.exception("job %s failed", job_id)
        raise HTTPException(500, f"inference failed: {exc}") from exc
    finally:
        shutil.rmtree(staging, ignore_errors=True)


_MEDIA = {
    ".glb": "model/gltf-binary",
    ".json": "application/json",
    ".gz": "application/gzip",
}


@app.get("/api/jobs/{job_id}/{filename}")
def asset(job_id: str, filename: str):
    if not job_id.isalnum() or "/" in filename or ".." in filename:
        raise HTTPException(400, "bad path")
    path = JOBS / job_id / filename
    if not path.is_file():
        raise HTTPException(404, f"{filename} not found for job {job_id}")
    return FileResponse(
        path,
        media_type=_MEDIA.get(path.suffix, "application/octet-stream"),
        filename=filename,
    )
