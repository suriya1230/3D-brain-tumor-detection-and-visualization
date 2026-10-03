"""
Segmentation inference for the 2026-10 GLI+MEN+PED retrain.

Replaced the earlier SegResNet/glioma-only model. This one is a DynUNet
(nnU-Net-style encoder-decoder, see backend/models/traning-notebook.ipynb
cell 10, "7.1 backbone (nnU-Net / DynUNet)") trained on three combined
BraTS cohorts - BraTS-GLI (adult glioma), BraTS-MEN (meningioma), BraTS-PED
(pediatric glioma) - with NO classification head and NO cohort output of
any kind: it segments the same four region channels (TC/WT/ET/RC)
regardless of which cohort a scan actually came from, and nothing in this
pipeline can tell you which one it was.

The raw model's output alone is substantially worse than the old
single-cohort model (pooled lesion-wise Dice ~0.53 on the 293-case held-out
test set, vs the old model's 0.80 single-cohort holdout) - multi-cohort
segmentation is a harder task. A learned per-lesion "component gate" (a
HistGradientBoostingClassifier, backend/models/component_gate.joblib)
recovers most of that gap post-hoc by rejecting predicted lesions that look
like false positives, bringing pooled Dice to ~0.71 - see _apply_gate()
below and the README's model-provenance section for the full numbers,
including the real trade-off the gate makes (it measurably costs recall on
the smallest lesions to buy that precision - this is not a free lunch).

Preprocessing chain (load -> concat -> RAS -> 1mm spacing -> crop
foreground -> normalise) is unchanged from the old model - the training
notebook treats that as already done by a separate "prep" step, not
something this retrain touched.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.networks.nets import DynUNet
from monai.transforms import (
    Compose,
    ConcatItemsd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    Spacingd,
)
from scipy import ndimage as ndi

log = logging.getLogger(__name__)

MODALITIES = ["t1n", "t1c", "t2w", "t2f"]
REGION_NAMES = ["TC", "WT", "ET", "RC"]      # model output channel order
PIXDIM = (1.0, 1.0, 1.0)
ROI = (96, 96, 96)

# Full 26-connectivity (face+edge+corner neighbours) - must match the
# training notebook's CONN26 = ndi.generate_binary_structure(3, 3) exactly,
# since the gate was trained on components labelled with this structure.
CONN26 = ndi.generate_binary_structure(3, 3)

# Exact order the component_gate classifier expects its feature vector in
# (backend/models/traning-notebook.ipynb cell 21, FEATURE_NAMES) - this is
# not cosmetic, predict_proba() has no idea what a column "means", only
# its position, so a reordering here would silently score the wrong thing
# rather than error out.
GATE_FEATURE_NAMES = [
    "region", "volume", "log_volume", "mean_prob", "max_prob", "std_prob",
    "prob_mass", "fill_ratio", "frac_of_largest", "n_comp", "touches_border",
]
# cfg.lw_min_pred in the training notebook - a component smaller than this
# is never kept regardless of what the gate scores it, same rule the
# notebook applies in apply_gate_to_labelmap(). Not persisted in
# gate_thresholds.json (only best_vol_thresh/best_gate_tau are), so it's
# pinned here from the training cfg dict instead.
GATE_MIN_PRED_VOXELS = 10

# Each additional TTA pass (see predict()) is a full extra sliding-window
# forward pass over the whole volume - on CPU this is the dominant cost of
# a request, so this is a real speed/uncertainty-signal tradeoff, not a
# free knob. 4 was the original "identity + one flip per axis" set; 2
# (identity + one flip) roughly halves inference time versus that while
# still measuring some self-disagreement. Set TTA_PASSES=1 to disable TTA
# entirely and go back to the original single-pass speed - region
# uncertainty then reports None (not 0.0 - 0.0 would claim "measured, no
# disagreement" when really nothing was measured at all).
_ALL_TTA_DIMS = [None, [2], [3], [4]]
TTA_PASSES = max(1, min(int(os.getenv("TTA_PASSES", "2")), len(_ALL_TTA_DIMS)))

# How many ROI windows sliding_window_inference runs through the model in
# one forward call. This is pure inference-time batching - mathematically
# identical output to looping one window at a time, just less per-call
# Python/op overhead - so unlike TTA_PASSES above, there is no accuracy
# tradeoff here, only a memory/speed one. Was hardcoded to 1 (one window
# at a time), which on a no-GPU host serialized every one of a volume's
# ~40-50 windows through the model individually. 4 is a conservative
# default for a CPU host with no GPU memory ceiling to worry about;
# raise it if RAM allows, or back to 1 if a request OOMs.
SW_BATCH_SIZE = max(1, int(os.getenv("SW_BATCH_SIZE", "4")))


@dataclass
class Prediction:
    seg: np.ndarray          # uint8 label map: 1 NETC, 2 SNFH, 3 ET, 4 RC
    prob: np.ndarray         # (4, X, Y, Z) float32 sigmoid, channels REGION_NAMES - raw, NOT gate-filtered
    affine: np.ndarray       # 4x4, voxel index -> world mm (RAS)
    t1n: np.ndarray          # resampled T1, same grid as seg (for the brain mesh)
    epoch: int
    val_dice: float          # voxel-wise Dice on the in-training validation subset (SELECT_REGIONS mean)
    lesion_dice: float       # lesion-wise Dice on the same subset - a different, harder metric (see module docstring)
    gate_applied: bool       # False if component_gate.joblib/gate_thresholds.json were missing and seg fell back to a raw 0.5 threshold
    region_uncertainty: dict[str, float | None]  # REGION_NAMES -> TTA disagreement, or
                                                   # None if the region predicted no voxels


# ---------------------------------------------------------------- model ----
def build_model(
    *, kernels, strides, deep_supr_num: int, in_channels: int, out_channels: int,
    dropout: float, deep_supervision: bool,
) -> DynUNet:
    """DynUNet (nnU-Net-style), matching backend/models/traning-notebook.ipynb
    cell 10 exactly. kernels/strides/deep_supr_num/in_channels/out_channels/
    dropout/deep_supervision are NOT hardcoded here - they're read from the
    checkpoint itself (see Segmenter.__init__) because they were derived at
    training time from cfg.roi/cfg.pixdim, and the saved state_dict's shape
    depends on them exactly (deep_supervision=True adds extra deep-
    supervision-head conv layers as real parameters - construct with a
    different value than training used and load_state_dict fails on a
    key/shape mismatch, it doesn't just silently work with different
    inference behaviour). model.eval() always returns a single tensor
    regardless of deep_supervision - the stacked multi-head output only
    appears in train mode, so this flag only affects what weights exist,
    not what a loaded-and-eval()'d model outputs.
    """
    return DynUNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernels,
        strides=strides,
        upsample_kernel_size=strides[1:],
        filters=None,
        norm_name=("INSTANCE", {"affine": True}),
        act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        deep_supervision=deep_supervision,
        deep_supr_num=deep_supr_num,
        dropout=(dropout if dropout and dropout > 0 else None),
        res_block=False,
    )


# ------------------------------------------------------ component gate -----
_GATE_CACHE: dict | None = None


def _load_gate(models_dir: Path) -> dict:
    """Loads component_gate.joblib + gate_thresholds.json once per process.
    Missing files degrade to "no gating" (raw 0.5 threshold) rather than
    crashing - but that's a real accuracy regression, not a harmless
    fallback (see module docstring's 0.53-vs-0.71 numbers), so it's logged
    as a warning, not silently absorbed.
    """
    global _GATE_CACHE
    if _GATE_CACHE is not None:
        return _GATE_CACHE

    gate_path = models_dir / "component_gate.joblib"
    thresh_path = models_dir / "gate_thresholds.json"
    if not gate_path.exists() or not thresh_path.exists():
        log.warning(
            "component gate not found (%s / %s) - falling back to a raw "
            "0.5 probability threshold with no per-lesion filtering. This "
            "is a real accuracy regression, not a cosmetic fallback: the "
            "shipped model's pooled lesion-wise Dice is ~0.53 ungated vs "
            "~0.71 gated on the held-out test set. Put both files in "
            "backend/models/ to fix this.",
            gate_path, thresh_path,
        )
        _GATE_CACHE = {"gate": None, "tau": None}
        return _GATE_CACHE

    d = joblib.load(gate_path)
    saved_features = list(d.get("features", []))
    if saved_features != GATE_FEATURE_NAMES:
        raise ValueError(
            f"component_gate.joblib's feature order {saved_features} != "
            f"this code's GATE_FEATURE_NAMES {GATE_FEATURE_NAMES} - "
            f"predict_proba() has no way to know columns were reordered, "
            f"it would silently score the wrong thing rather than error, "
            f"so this is a hard failure instead of a warning."
        )
    thresholds = json.loads(thresh_path.read_text(encoding="utf-8"))
    _GATE_CACHE = {"gate": d["model"], "tau": float(thresholds["best_gate_tau"])}
    log.info("loaded component gate (tau=%.2f)", _GATE_CACHE["tau"])
    return _GATE_CACHE


def _region_components(prob_channel: np.ndarray, region_idx: int) -> list[dict]:
    """Connected-component feature extraction for one region channel,
    ported from region_components() (training notebook cell 13).

    Inference-only simplification: the notebook's version also matches
    components against ground truth to compute TP/FP/FN for evaluation.
    That half is dropped here, not approximated - there is no ground truth
    at inference time, and the notebook's own apply_gate_to_labelmap()
    always calls region_components() with gt_bin as an all-zero array, which
    makes the ground-truth-matching code a provable no-op in exactly this
    call pattern (an empty ground truth produces zero matched lesions no
    matter what), not a behaviour change.
    """
    pred_bin = prob_channel > 0.5
    lab, n = ndi.label(pred_bin, structure=CONN26)
    if not n:
        return []

    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    idx = list(range(1, n + 1))
    mean_prob = np.atleast_1d(ndi.mean(prob_channel, lab, idx))
    max_prob = np.atleast_1d(ndi.maximum(prob_channel, lab, idx))
    std_prob = np.atleast_1d(ndi.standard_deviation(prob_channel, lab, idx))
    boxes = ndi.find_objects(lab)
    largest = float(max(sizes[1:]))

    comps = []
    for k, j in enumerate(idx):
        vol = int(sizes[j])
        box = boxes[j - 1]
        bvol, border = 1, 0
        for ax, sl in enumerate(box):
            bvol *= (sl.stop - sl.start)
            if sl.start == 0 or sl.stop == prob_channel.shape[ax]:
                border = 1
        mp = float(mean_prob[k])
        comps.append({
            "id": j,
            "volume": vol,
            "f": {
                "region": float(region_idx),
                "volume": float(vol),
                "log_volume": float(np.log1p(vol)),
                "mean_prob": mp,
                "max_prob": float(max_prob[k]),
                "std_prob": float(std_prob[k]),
                "prob_mass": mp * vol,
                "fill_ratio": float(vol / max(bvol, 1)),
                "frac_of_largest": float(vol / max(largest, 1.0)),
                "n_comp": float(n),
                "touches_border": float(border),
            },
        })
    return comps


def _apply_gate(prob: np.ndarray, gate, tau: float | None) -> np.ndarray:
    """Per-region connected-component gating: keep a predicted lesion only
    if the learned classifier scores it >= tau AND it's at least
    GATE_MIN_PRED_VOXELS voxels. Ported from apply_gate_to_labelmap()
    (training notebook cell 29). Runs on `prob` in whatever grid it's
    given - predict() below calls this on the CROPPED grid (matching the
    notebook's own predict_case(), which gates directly on the sliding-
    window output before any pasting into a larger canvas), so
    "touches_border" means touches the edge of the cropped foreground
    region, not the edge of the original uncropped scan.
    """
    kept = np.zeros_like(prob, dtype=bool)
    for c in range(prob.shape[0]):
        comps = _region_components(prob[c], region_idx=c)
        binm = prob[c] > 0.5
        if not comps or gate is None or tau is None:
            kept[c] = binm
            continue
        lab, _ = ndi.label(binm, structure=CONN26)
        X = np.asarray(
            [[cc["f"].get(k, 0.0) for k in GATE_FEATURE_NAMES] for cc in comps],
            dtype=np.float32,
        )
        p = gate.predict_proba(X)[:, 1]
        keep_ids = [
            cc["id"] for cc, pi in zip(comps, p)
            if pi >= tau and cc["volume"] >= GATE_MIN_PRED_VOXELS
        ]
        kept[c] = np.isin(lab, keep_ids) if keep_ids else np.zeros_like(binm)
    return kept


class Segmenter:
    """Loads the checkpoint once and keeps it warm for the process lifetime."""

    def __init__(self, ckpt_path: str | Path, device: str | None = None):
        self.ckpt_path = Path(ckpt_path)
        if not self.ckpt_path.exists():
            raise FileNotFoundError(
                f"checkpoint not found: {self.ckpt_path}\n"
                f"Put best_lesion.pt (or best_weights.pt / best.pt) in backend/models/."
            )

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if self.device.type == "cpu":
            # Torch's own default (seen on this host: 4 of 8 cores) is
            # tuned to leave room for other processes on a shared
            # machine, not to minimize one request's latency - the wrong
            # default for a backend whose whole job IS this inference
            # call. Every intra-op (conv, etc.) in the sliding-window
            # loop benefits from the extra cores; only matters on CPU,
            # since a CUDA device ignores this setting entirely.
            torch.set_num_threads(os.cpu_count() or 1)
        ck = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)

        saved = ck.get("region_names")
        if saved and list(saved) != REGION_NAMES:
            raise ValueError(
                f"checkpoint regions {saved} != expected {REGION_NAMES}; "
                f"the label mapping differs, so this checkpoint cannot be used"
            )

        kernels = ck.get("kernels")
        strides = ck.get("strides")
        if not kernels or not strides:
            raise ValueError(
                f"{self.ckpt_path.name} has no saved kernels/strides - this "
                f"code builds a DynUNet matching backend/models/traning-"
                f"notebook.ipynb and needs those saved in the checkpoint; "
                f"an older SegResNet-era checkpoint is not compatible with "
                f"this version of inference.py."
            )
        cfg = ck.get("cfg") or {}

        self.model = build_model(
            kernels=kernels,
            strides=strides,
            deep_supr_num=int(ck.get("deep_supr_num", 2)),
            in_channels=int(cfg.get("in_channels", 4)),
            out_channels=int(cfg.get("out_channels", 4)),
            dropout=float(cfg.get("dropout", 0.0)),
            deep_supervision=bool(cfg.get("deep_supervision", True)),
        )
        state = ck["model_state"] if "model_state" in ck else ck
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

        self.epoch = int(ck.get("epoch", -1)) + 1
        self.val_dice = float(ck.get("best_metric", float("nan")))
        self.lesion_dice = float(ck.get("best_lesion", float("nan")))

        gate_info = _load_gate(self.ckpt_path.parent)
        self.gate = gate_info["gate"]
        self.gate_tau = gate_info["tau"]

        self.pre = Compose([
            LoadImaged(keys=MODALITIES, image_only=False),
            EnsureChannelFirstd(keys=MODALITIES),
            ConcatItemsd(keys=MODALITIES, name="image", dim=0),
            EnsureTyped(keys="image"),
            Orientationd(keys="image", axcodes="RAS"),
            Spacingd(keys="image", pixdim=PIXDIM, mode="bilinear"),
        ])

        log.info(
            "loaded %s (epoch %d, val Dice %.4f, lesion Dice %.4f, gate %s) on %s",
            self.ckpt_path.name, self.epoch, self.val_dice, self.lesion_dice,
            "on" if self.gate is not None else "OFF (missing files)", self.device,
        )

    # ------------------------------------------------------------------
    @torch.inference_mode()  # stricter + slightly cheaper than no_grad - no
    # autograd version-counter bookkeeping at all, and nothing here needs
    # a tensor to re-enter a graph afterward (every output is immediately
    # pulled out to numpy).
    def predict(self, paths: dict[str, str], sw_overlap: float = 0.5) -> Prediction:
        """paths maps each of t1n/t1c/t2w/t2f to a NIfTI file path."""
        missing = [m for m in MODALITIES if m not in paths]
        if missing:
            raise ValueError(f"missing modalities: {missing}")

        data = self.pre({m: str(paths[m]) for m in MODALITIES})
        img = data["image"]                                  # (4, X, Y, Z)
        affine = np.asarray(img.affine, dtype=np.float64)
        full = torch.as_tensor(np.asarray(img)).float()
        shape = tuple(full.shape[1:])

        # --- foreground crop, matching CropForegroundd(source_key="image") ---
        fg = (full > 0).any(dim=0).numpy()
        if not fg.any():
            raise ValueError(
                "all four inputs are empty after resampling - are these really "
                "brain MRI volumes?"
            )
        idx = np.argwhere(fg)
        lo = idx.min(axis=0)
        hi = idx.max(axis=0) + 1
        crop = full[:, lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].clone()

        # --- normalise after cropping, per channel, ignoring zeros -----------
        crop = NormalizeIntensityd(
            keys="image", nonzero=True, channel_wise=True
        )({"image": crop})["image"]
        crop = torch.as_tensor(np.asarray(crop)).float()

        # --- pad up to the ROI if the brain is smaller than one window ------
        pad, cs = [], list(crop.shape[1:])
        for d in range(3):
            need = max(0, ROI[d] - cs[d])
            pad = [0, need] + pad          # F.pad wants reversed axis order
        if any(pad):
            crop = torch.nn.functional.pad(crop, pad)

        log.info("input %s -> cropped %s", shape, tuple(crop.shape[1:]))

        x = crop.unsqueeze(0).to(self.device)
        use_amp = self.device.type == "cuda"

        # --- test-time augmentation: identity + up to 3 axis flips ----------
        # TTA_PASSES (module constant) controls how many of these actually
        # run - each one is a full extra forward pass, the dominant cost of
        # a request on CPU. Flip is applied to the already-padded tensor and
        # undone on the logits before anything else touches them, so the pad
        # region still lands in the same place for every pass. Comparing the
        # outputs measures how much the model disagrees with itself under a
        # symmetry it should in principle be invariant to - that
        # disagreement is a per-region uncertainty *signal*, not a
        # calibrated error probability (see meshing.py's meta["confidence"]
        # for the honesty caveat, matching how this project already treats
        # mean_probability).
        tta_dims = _ALL_TTA_DIMS[:TTA_PASSES]
        passes = []
        for dims in tta_dims:
            xin = torch.flip(x, dims=dims) if dims else x
            with torch.autocast(self.device.type, enabled=use_amp):
                logits = sliding_window_inference(
                    xin, ROI, sw_batch_size=SW_BATCH_SIZE, predictor=self.model,
                    overlap=sw_overlap, mode="gaussian", progress=False,
                )
            if dims:
                logits = torch.flip(logits, dims=dims)
            passes.append(torch.sigmoid(logits.float())[0].cpu().numpy())

        stack = np.stack(passes, axis=0)   # (4, 4, x, y, z): passes, regions, ...
        p = stack.mean(axis=0)
        p_std = stack.std(axis=0)

        # --- undo pad -----------------------------------------------------
        cs = [hi[d] - lo[d] for d in range(3)]
        p = p[:, : cs[0], : cs[1], : cs[2]]
        p_std = p_std[:, : cs[0], : cs[1], : cs[2]]

        # --- per-lesion gating, on the cropped grid (see _apply_gate) ------
        kept = _apply_gate(p, self.gate, self.gate_tau)

        # --- paste both the raw probabilities and the gated mask back ------
        prob = np.zeros((4, *shape), dtype=np.float32)
        prob[:, lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = p

        kept_full = np.zeros((4, *shape), dtype=bool)
        kept_full[:, lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = kept

        seg = regions_to_labelmap(kept_full)
        t1n = np.asarray(full[0].numpy(), dtype=np.float32)

        region_uncertainty: dict[str, float | None] = {}
        for i, name in enumerate(REGION_NAMES):
            region_mask = p[i] > 0.5
            # Fewer than 2 passes means nothing was actually compared -
            # None, not a std of 0.0 computed from a single sample, which
            # would misreport "no disagreement" as if it had been measured.
            if len(tta_dims) < 2:
                region_uncertainty[name] = None
            else:
                region_uncertainty[name] = (
                    float(p_std[i][region_mask].mean()) if region_mask.any() else None
                )

        return Prediction(
            seg=seg, prob=prob, affine=affine, t1n=t1n,
            epoch=self.epoch, val_dice=self.val_dice, lesion_dice=self.lesion_dice,
            gate_applied=self.gate is not None,
            region_uncertainty=region_uncertainty,
        )


# ------------------------------------------------------- label assembly ----
def regions_to_labelmap(binary: np.ndarray) -> np.ndarray:
    """
    (4, X, Y, Z) region masks (TC, WT, ET, RC) -> BraTS label map.

    Regions are predicted independently, so they can disagree: a voxel may be
    ET without being TC. Enforce the nesting ET <= TC <= WT before assigning
    labels, otherwise the label map contradicts the region volumes.
    RC is surgical and sits outside WT, so it never overwrites tumour.
    """
    tc, wt, et, rc = (binary[i].astype(bool) for i in range(4))
    tc = tc | et
    wt = wt | tc

    seg = np.zeros(binary.shape[1:], dtype=np.uint8)
    seg[wt & ~tc] = 2        # SNFH
    seg[tc & ~et] = 1        # NETC
    seg[et] = 3              # ET
    seg[rc & (seg == 0)] = 4  # RC
    return seg
