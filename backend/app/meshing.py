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

log = logging.getLogger(__name__)

# label value -> (node name, default visible in the viewer)
REGIONS: dict[int, tuple[str, bool]] = {
    3: ("ET", True),        # enhancing tissue
    1: ("NETC", True),      # non-enhancing tumour core
    4: ("RC", True),        # resection cavity
    2: ("SNFH", False),     # FLAIR hyperintensity: large, hides everything else
}

PROB_CHANNEL = {"TC": 0, "WT": 1, "ET": 2, "RC": 3}

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
    """BraTS T1 is skull-stripped, so background is exactly 0."""
    m = t1 > 0
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

    meta = {
        "case_id": case_id,
        "model": "SegResNet (MONAI), BraTS 2024 post-treatment glioma",
        "checkpoint": f"epoch {pred.epoch}",
        "validation_select_dice": round(pred.val_dice, 4),
        "holdout_select_dice": 0.8018,
        "voxel_volume_mm3": vox_mm3,
        "space": "MNI152 1 mm" if lobe_of else "native",
        "scene_radius_mm": round(radius, 1),
        "brain_faces": int(len(brain_mesh.faces)),
        "regions": regions,
        "derived": derived,
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
