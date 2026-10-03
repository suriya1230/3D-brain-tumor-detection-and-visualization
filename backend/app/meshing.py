"""
Prediction -> web-ready GLB meshes plus a metadata dict.

Two conventions the frontend depends on:

  * Axis map (x, y, z) -> (x, z, -y). RAS has z up, three.js has y up. This
    permutation preserves handedness, so nothing renders mirrored. A mirrored
    brain puts the tumour in the wrong hemisphere, which is not a cosmetic bug.
  * Both meshes are centred on the *brain* centroid, never on their own. That
    is what keeps the tumour in its real position inside the brain.
"""

from __future__ import annotations

import logging

import nibabel as nib
import numpy as np
import trimesh
from scipy import ndimage
from skimage import measure
from skimage.filters import threshold_otsu

from app.inference import TTA_PASSES

log = logging.getLogger(__name__)

# label value -> (node name, default visible in the viewer)
REGIONS: dict[int, tuple[str, bool]] = {
    3: ("ET", True),        # enhancing tissue
    1: ("NETC", True),      # non-enhancing tumour core
    4: ("RC", True),        # resection cavity
    2: ("SNFH", False),     # FLAIR hyperintensity: large, hides everything else
}

PROB_CHANNEL = {"TC": 0, "WT": 1, "ET": 2, "RC": 3}

# Lesion-wise Dice per region on the 293-case held-out test set, WITH the
# shipped component gate applied (gate tau=0.6, single global threshold -
# the variant actually persisted in gate_thresholds.json), pooled across
# all three training cohorts (BraTS-GLI/MEN/PED) - see
# backend/models/traning-notebook.ipynb cell 25. This replaced a
# single-cohort (glioma-only) model's 0.80 holdout Dice; multi-cohort
# segmentation is measurably harder, and the gate is not optional
# overhead - without it, pooled Dice drops to ~0.53 (see
# REGION_HOLDOUT_DICE_UNGATED below).
REGION_HOLDOUT_DICE = {"TC": 0.7045, "WT": 0.7504, "ET": 0.6768, "RC": 0.4584}

# Same metric, same checkpoint, gate turned off (raw 0.5 threshold) - kept
# alongside the gated numbers rather than discarded, because the gap
# between them (0.53 vs 0.71 pooled) IS the headline fact about this
# model: the gate is doing real work, not a marginal tweak, and a reader
# should be able to see both an pipelines a result actually depends on.
REGION_HOLDOUT_DICE_UNGATED = {"TC": 0.5548, "WT": 0.5062, "ET": 0.5282, "RC": 0.2622}

# Per-cohort pooled Dice (mean of TC/WT/ET, gate applied) - performance
# varies a lot by which of the three training cohorts a case resembles,
# and there is no classification head anywhere in this pipeline: nothing
# can tell you which regime a given scan is actually in. Surfaced as a
# range, not averaged away, for the same reason inference.py's per-region
# TTA uncertainty is reported per-region instead of as one case-level
# number - a single pooled figure hides exactly the information a reader
# would need to calibrate trust.
COHORT_HOLDOUT_DICE = {
    "BraTS-GLI (adult glioma)": 0.6354,
    "BraTS-MEN (meningioma)": 0.8065,
    "BraTS-PED (pediatric glioma)": 0.6292,
}

# RC (resection cavity) ground truth exists ONLY in the BraTS-GLI cohort -
# BraTS-MEN and BraTS-PED's own label protocols don't define a resection
# cavity class at all (confirmed empirically: test_lesionwise.csv shows
# zero RC ground-truth instances, fn=0, for both cohorts in the full test
# set). The model has no cohort-awareness, so it still emits an RC channel
# for every case regardless of cohort - for a MEN or PED scan, that output
# was never checked against any ground truth during training or
# evaluation. Not a documented caveat in the training notebook itself; see
# README's model-provenance section for where this is actually said.
RC_VALIDATED_COHORTS_ONLY = "BraTS-GLI (adult glioma)"

BRAIN_STEP = 2      # marching-cubes stride for the brain (2 keeps it light)
TUMOUR_STEP = 1     # full resolution for tumour regions
SMOOTH_SIGMA = 1.0
TAUBIN_ITERS = 12
MIN_VOXELS = 20     # below this a region is noise, not a finding


def _surface(mask, affine, step, sigma=SMOOTH_SIGMA, taubin=TAUBIN_ITERS):
    """Binary mask -> trimesh in the affine's world space (mm)."""
    f = mask.astype(np.float32)
    if sigma:
        f = ndimage.gaussian_filter(f, sigma)
    if f.max() <= 0.5:
        return None
    verts, faces, _, _ = measure.marching_cubes(f, level=0.5, step_size=step)
    verts = nib.affines.apply_affine(affine, verts)
    m = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    m.remove_degenerate_faces()
    m.remove_unreferenced_vertices()
    if taubin:
        m = trimesh.smoothing.filter_taubin(m, iterations=taubin)
    m.fix_normals()
    return m


def _brain_mask(t1: np.ndarray) -> np.ndarray:
    """Was a bare `t1 > 0` on the assumption that BraTS T1 is always
    skull-stripped to an exact zero background. That held for BraTS-GLI
    (confirmed: a GLI case's brain mesh filled 47% of its own bounding
    box - a normal organic brain shape) but not for BraTS-PED (confirmed:
    a PED case through the same code filled 91% of its bounding box, a
    near-solid block, ~5081 cc vs. a real brain's ~1200-1500 cc, with a
    227mm top-to-bottom extent - consistent with neck/shoulder tissue
    still attached rather than a clean skull-strip). This cohort
    difference is pre-existing in the training data, not something this
    retrain introduced - BraTS-PED and BraTS-MEN simply aren't guaranteed
    to ship with the same zero-background convention as BraTS-GLI.

    Otsu threshold on the nonzero intensity histogram, instead of a bare
    `> 0` - adapts to a noisy/non-zero background automatically rather
    than assuming it's exactly zero. Confirmed by direct shaded-render
    inspection to produce a clean, undamaged brain-shaped mesh.

    An erosion-based "neck cut" (sever a thin neck by eroding, keep the
    largest post-erosion component, dilate back by the same fixed amount)
    was tried here and reverted - confirmed by that same direct shaded
    render to carve real gouges and faceted flat plateaus into the actual
    brain surface, not just trim the neck. A real brain has many deep,
    narrow sulcal folds; a fixed dilation count can't regrow into a fold
    that needs more steps than the neck's own width without ALSO
    regrowing back through the neck, so there's no single iteration count
    that restores genuine anatomy without either leaving gouges or
    undoing the cut. That's a worse failure than the oversized shell it
    was meant to fix - a slightly-too-large decorative brain reads as "not
    perfectly cropped"; a gouged one reads as "the segmentation is
    broken." Properly solving the neck case (BraTS-PED/MEN scans that
    include extra-cranial tissue) would need real sample data from that
    cohort to tune and validate a cut against, which this environment
    doesn't have (the original T1 upload isn't retained after a job
    completes) - until then, those cohorts' decorative brain shell may be
    visibly larger/more elongated than BraTS-GLI's, which is an honest
    cosmetic limitation, not a damaged one.
    """
    nz = t1[t1 > 0]
    thresh = threshold_otsu(nz) if nz.size else 0
    m = t1 > thresh
    m = ndimage.binary_closing(m, iterations=2)
    m = ndimage.binary_fill_holes(m)
    lab, n = ndimage.label(m)
    if n > 1:                                    # keep the largest component
        m = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
    return m


def _lobe_namer(shape):
    """
    Coarse anatomical naming from MNI152 coordinates.

    Only offered when the grid is MNI152 1 mm (182x218x182), which is what
    BraTS 2024 GLI ships in. This is a set of thresholds, not an atlas
    segmentation: good enough for 'left frontal', not for a gyrus.
    """
    if tuple(shape) != (182, 218, 182):
        return None

    def name(mni):
        x, y, z = mni
        side = "left" if x < -4 else "right" if x > 4 else "midline"
        if y > 18:
            part = "frontal lobe"
        elif y < -58:
            part = "occipital lobe"
        elif z < -2:
            part = "temporal lobe"
        elif z > 28:
            part = "parietal lobe"
        else:
            part = "central region"
        return f"{side} {part}"

    return name


def build_assets(pred, case_id: str) -> tuple[bytes, bytes, dict]:
    """Returns (brain_glb, tumor_glb, metadata)."""
    brain = _brain_mask(pred.t1n)
    brain_mesh = _surface(brain, pred.affine, BRAIN_STEP)
    if brain_mesh is None:
        raise ValueError("brain surface came out empty - check the T1 input")

    centre = brain_mesh.vertices.mean(axis=0).copy()
    radius = float(np.linalg.norm(brain_mesh.vertices - centre, axis=1).max())

    def to_scene_frame(m):
        v = m.vertices - centre
        m.vertices = np.column_stack([v[:, 0], v[:, 2], -v[:, 1]])
        m.fix_normals()
        return m

    brain_mesh = to_scene_frame(brain_mesh)

    vox_mm3 = float(abs(np.linalg.det(pred.affine[:3, :3])))
    lobe_of = _lobe_namer(pred.seg.shape)

    tumour_scene = trimesh.Scene()
    regions: dict[str, dict] = {}

    for value, (name, visible) in REGIONS.items():
        mask = pred.seg == value
        n_vox = int(mask.sum())
        if n_vox < MIN_VOXELS:
            log.info("%s: absent (%d voxels)", name, n_vox)
            continue

        mesh = _surface(mask, pred.affine, TUMOUR_STEP)
        if mesh is None:
            continue

        centroid = nib.affines.apply_affine(
            pred.affine, np.argwhere(mask).mean(axis=0)
        )
        entry = {
            "volume_cc": round(n_vox * vox_mm3 / 1000.0, 2),
            "voxels": n_vox,
            "faces": int(len(mesh.faces)),
            "default_visible": visible,
            "centroid_mm": [round(float(c), 1) for c in centroid],
        }
        if lobe_of is not None:
            entry["location"] = lobe_of(centroid)

        ch = PROB_CHANNEL.get(name)
        if ch is not None:
            entry["mean_probability"] = round(float(pred.prob[ch][mask].mean()), 4)
            unc = (pred.region_uncertainty or {}).get(name)
            if unc is not None:
                entry["tta_uncertainty"] = round(unc, 4)

        tumour_scene.add_geometry(to_scene_frame(mesh), geom_name=name,
                                  node_name=name)
        regions[name] = entry
        log.info("%s: %.2f cc, %d faces", name, entry["volume_cc"],
                 entry["faces"])

    if not regions:
        raise ValueError(
            "no tumour regions above threshold - the model found nothing. "
            "Check that the four inputs are the correct modalities."
        )

    derived: dict[str, float] = {}
    tc = [k for k in ("NETC", "ET") if k in regions]
    wt = [k for k in ("NETC", "SNFH", "ET") if k in regions]
    if tc:
        derived["TC_cc"] = round(sum(regions[k]["volume_cc"] for k in tc), 2)
    if wt:
        derived["WT_cc"] = round(sum(regions[k]["volume_cc"] for k in wt), 2)
    if "ET" in regions and derived.get("WT_cc"):
        derived["ET_WT_ratio"] = round(
            regions["ET"]["volume_cc"] / derived["WT_cc"], 3
        )

    region_uncertainty = {
        k: round(v, 4) for k, v in (pred.region_uncertainty or {}).items() if v is not None
    }

    meta = {
        "case_id": case_id,
        "model": (
            "DynUNet (MONAI, nnU-Net-style), trained on BraTS-GLI (adult "
            "glioma) + BraTS-MEN (meningioma) + BraTS-PED (pediatric "
            "glioma) combined. No classification head - this model cannot "
            "tell you which of the three cohorts a scan actually "
            "resembles, and its segmentation accuracy differs substantially "
            "between them (see confidence.cohort_holdout_dice)."
        ),
        "checkpoint": f"epoch {pred.epoch}",
        # Two different metrics on two different, non-overlapping splits -
        # keep both the numbers AND their set sizes attached, so nothing
        # downstream can quietly present the 60-case validation-time voxel
        # score as if it were measured on the same 293-case held-out test
        # set as the lesion-wise number (see phase1_bridge.py, which reads
        # these four fields together rather than letting holdout Dice
        # stand alone as "the" accuracy figure).
        "validation_select_dice": round(pred.val_dice, 4),
        "validation_set_size": 60,
        "validation_lesion_dice": round(pred.lesion_dice, 4),
        "holdout_select_dice": 0.7106,
        "holdout_set_size": 293,
        "component_gate_applied": pred.gate_applied,
        "voxel_volume_mm3": vox_mm3,
        "space": "MNI152 1 mm" if lobe_of else "native",
        "scene_radius_mm": round(radius, 1),
        "brain_faces": int(len(brain_mesh.faces)),
        "regions": regions,
        "derived": derived,
        "confidence": {
            "method": (
                f"Test-time augmentation: {TTA_PASSES} pass(es) (identity "
                f"+ one flip per spatial axis, up to 3). tta_uncertainty "
                f"is the mean per-voxel disagreement (probability std "
                f"across passes) inside each predicted region for THIS "
                f"scan."
            ),
            "region_uncertainty": region_uncertainty,
            "region_holdout_dice": REGION_HOLDOUT_DICE,
            "region_holdout_dice_ungated": REGION_HOLDOUT_DICE_UNGATED,
            "cohort_holdout_dice": COHORT_HOLDOUT_DICE,
            "rc_validated_cohorts_only": RC_VALIDATED_COHORTS_ONLY,
            "note": (
                "tta_uncertainty is a raw disagreement score, not a "
                "calibrated error probability - no per-case holdout data "
                "exists yet to calibrate it against ground truth. Higher "
                "means the model disagrees with itself more under a "
                "symmetry it should be invariant to; treat it as a relative "
                "signal within this case, not an absolute accuracy figure. "
                "region_holdout_dice is with the component gate applied "
                "(the pipeline this case actually ran through); "
                "region_holdout_dice_ungated is the same model with the "
                "gate turned off, shown alongside it because the gap "
                "between them is the headline fact about this model, not a "
                "footnote. cohort_holdout_dice is reported per training "
                "cohort because there is no way to know which cohort this "
                "scan resembles - a single pooled number would hide that "
                "performance ranges from 0.63 to 0.81 depending on which "
                "regime a case is actually in."
            ),
        },
        "notes": (
            "Locations are coarse MNI-coordinate heuristics, not atlas "
            "segmentations. mean_probability is the average sigmoid output "
            "inside each predicted region and is not a calibrated confidence. "
            "Volumes are measured in 1 mm resampled space. Research use only."
        ),
    }

    brain_glb = trimesh.Scene({"brain": brain_mesh}).export(file_type="glb")
    tumor_glb = tumour_scene.export(file_type="glb")
    return brain_glb, tumor_glb, meta


def save_mask_nifti(pred, path) -> None:
    nib.save(nib.Nifti1Image(pred.seg, pred.affine), str(path))
