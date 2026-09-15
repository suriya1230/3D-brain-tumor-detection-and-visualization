"""
SegResNet inference, replicating the training notebook's val/test chain exactly.

Chain order matters and is not negotiable:
    load -> concat -> RAS -> 1mm spacing -> crop foreground -> normalise
The notebook cropped the foreground before normalising. Anything else is a
different preprocessing pipeline and the reported metrics no longer apply.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.networks.nets import SegResNet
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

log = logging.getLogger(__name__)

MODALITIES = ["t1n", "t1c", "t2w", "t2f"]
REGION_NAMES = ["TC", "WT", "ET", "RC"]      # model output channel order
PIXDIM = (1.0, 1.0, 1.0)
ROI = (96, 96, 96)


@dataclass
class Prediction:
    seg: np.ndarray          # uint8 label map: 1 NETC, 2 SNFH, 3 ET, 4 RC
    prob: np.ndarray         # (4, X, Y, Z) float32 sigmoid, channels REGION_NAMES
    affine: np.ndarray       # 4x4, voxel index -> world mm (RAS)
    t1n: np.ndarray          # resampled T1, same grid as seg (for the brain mesh)
    epoch: int
    val_dice: float
    region_uncertainty: dict[str, float | None]  # REGION_NAMES -> TTA disagreement, or
                                                   # None if the region predicted no voxels


# ---------------------------------------------------------------- model ----
def build_model() -> SegResNet:
    """Exactly the architecture from notebook cell 6.1."""
    return SegResNet(
        spatial_dims=3,
        init_filters=32,
        in_channels=4,
        out_channels=4,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        dropout_prob=0.2,
    )


class Segmenter:
    """Loads the checkpoint once and keeps it warm for the process lifetime."""

    def __init__(self, ckpt_path: str | Path, device: str | None = None):
        self.ckpt_path = Path(ckpt_path)
        if not self.ckpt_path.exists():
            raise FileNotFoundError(
                f"checkpoint not found: {self.ckpt_path}\n"
                f"Put best_weights.pt (or best.pt) in backend/models/."
            )

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        ck = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)

        saved = ck.get("region_names")
        if saved and list(saved) != REGION_NAMES:
            raise ValueError(
                f"checkpoint regions {saved} != expected {REGION_NAMES}; "
                f"the label mapping differs, so this checkpoint cannot be used"
            )

        self.model = build_model()
        state = ck["model_state"] if "model_state" in ck else ck
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

        self.epoch = int(ck.get("epoch", -1)) + 1
        self.val_dice = float(ck.get("best_metric", float("nan")))

        self.pre = Compose([
            LoadImaged(keys=MODALITIES, image_only=False),
            EnsureChannelFirstd(keys=MODALITIES),
            ConcatItemsd(keys=MODALITIES, name="image", dim=0),
            EnsureTyped(keys="image"),
            Orientationd(keys="image", axcodes="RAS"),
            Spacingd(keys="image", pixdim=PIXDIM, mode="bilinear"),
        ])

        log.info(
            "loaded %s (epoch %d, val Dice %.4f) on %s",
            self.ckpt_path.name, self.epoch, self.val_dice, self.device,
        )

    # ------------------------------------------------------------------
    @torch.no_grad()
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

        # --- test-time augmentation: identity + one flip per spatial axis --
        # 4 passes, not the full 8-way combination, so this stays tractable
        # on CPU inference. Flip is applied to the already-padded tensor and
        # undone on the logits before anything else touches them, so the pad
        # region still lands in the same place for every pass. Comparing the
        # 4 outputs measures how much the model disagrees with itself under
        # a symmetry it should in principle be invariant to - that
        # disagreement is a per-region uncertainty *signal*, not a
        # calibrated error probability (see meshing.py's meta["confidence"]
        # for the honesty caveat, matching how this project already treats
        # mean_probability).
        tta_dims = [None, [2], [3], [4]]
        passes = []
        for dims in tta_dims:
            xin = torch.flip(x, dims=dims) if dims else x
            with torch.autocast(self.device.type, enabled=use_amp):
                logits = sliding_window_inference(
                    xin, ROI, sw_batch_size=1, predictor=self.model,
                    overlap=sw_overlap, mode="gaussian", progress=False,
                )
            if dims:
                logits = torch.flip(logits, dims=dims)
            passes.append(torch.sigmoid(logits.float())[0].cpu().numpy())

        stack = np.stack(passes, axis=0)   # (4, 4, x, y, z): passes, regions, ...
        p = stack.mean(axis=0)
        p_std = stack.std(axis=0)

        # --- undo pad, then paste back into the full grid -------------------
        cs = [hi[d] - lo[d] for d in range(3)]
        p = p[:, : cs[0], : cs[1], : cs[2]]
        p_std = p_std[:, : cs[0], : cs[1], : cs[2]]

        prob = np.zeros((4, *shape), dtype=np.float32)
        prob[:, lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = p

        seg = regions_to_labelmap(prob > 0.5)
        t1n = np.asarray(full[0].numpy(), dtype=np.float32)

        region_uncertainty: dict[str, float | None] = {}
        for i, name in enumerate(REGION_NAMES):
            region_mask = p[i] > 0.5
            region_uncertainty[name] = (
                float(p_std[i][region_mask].mean()) if region_mask.any() else None
            )

        return Prediction(
            seg=seg, prob=prob, affine=affine, t1n=t1n,
            epoch=self.epoch, val_dice=self.val_dice,
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
