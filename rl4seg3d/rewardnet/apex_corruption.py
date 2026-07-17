"""Anatomically-plausible apex morphological corruptions for the apex reward-net dataset.

Philosophy (region approach, mirroring the *anatomical* reward-net pipeline): take a
CLEAN reference segmentation, apply a smooth morphological deformation localized to the
LV apex, and keep the result as `pred` while the clean seg is `gt`. The reward target is
then the standard dense agreement map ``y = (gt == pred)`` used by ``RewardNet3DDataset``
(``reward_unet_3d_datamodule.py:46``) -- automatically localized to the apex.

Unlike a naive pixel-band edit, deformations here move the endocardium/epicardium
*borders* smoothly. Each border is displaced along its normal by a smooth, apex-weighted
amount implemented as a spatially-varying threshold on the signed distance field (SDF).
The result is always a topologically valid mask (LV cavity inside a myocardial ring,
largest component only, holes filled) with smooth, anatomically plausible borders.

Modes (all apex-localized):
  * ``balloon_apex``  -- inflate & push the whole apical tip outward (endo + epi grow).
  * ``shrink_apex``   -- rounded foreshortening: the tip is pulled inward (endo + epi shrink).
  * ``thin_myo``      -- move the endocardium outward toward the epicardium -> thin apical wall.
  * ``thick_myo``     -- move the endocardium inward -> thick apical wall.
  * ``aneurysm``      -- cavity balloons out while the wall thins (endo grows a lot, epi a little).

Conventions:
  * Labels (``vital.data.camus.config.Label``): BG=0, LV=1, MYO=2, ATRIUM=3.
  * A 2-D frame is ``(H, W)`` int labels; a volume is ``(H, W, frames)`` (frames last).
    The reward dataset stores volumes as ``(1, H, W, frames)``; helpers accept either and
    squeeze a leading singleton channel.
  * The SAME mode and magnitude are applied to every frame of a volume (temporally
    consistent); only the apex position is re-tracked per frame so the deformation
    follows the moving tip.
  * ``mag_mm`` is the border displacement at the apex, in millimetres.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from vital.data.camus.config import Label

MODES = ("balloon_apex", "shrink_apex", "thin_myo", "thick_myo", "aneurysm")

# Per-mode apex border-displacement ranges (mm). Wall (myo) modes use a smaller range
# since myocardial thickness is only a few mm; cavity/tip modes can move further.
MODE_MAG_RANGES = {
    "balloon_apex": (4.0, 9.0),
    "shrink_apex": (4.0, 9.0),
    "thin_myo": (2.0, 5.0),
    "thick_myo": (2.0, 5.0),
    "aneurysm": (4.0, 9.0),
}


def sample_mode_and_mag(rng, mode="random"):
    """Pick a (mode, mag_mm) pair; mode chosen uniformly when ``mode == "random"``."""
    chosen = rng.choice(MODES) if mode == "random" else mode
    lo, hi = MODE_MAG_RANGES[chosen]
    return chosen, float(rng.uniform(lo, hi))


def _apex_and_axis(seg2d, lv=Label.LV, myo=Label.MYO):
    """Return (apex, base_mid, unit_axis_toward_apex, length_px), orientation-free.

    base_mid = centroid of the LV boundary that touches background (the open valve
    plane; the cavity is myocardium-capped everywhere else); apex = LV pixel farthest
    from base_mid. Raises RuntimeError on a degenerate axis; callers skip the frame.
    """
    lv_mask = seg2d == int(lv)
    bg = ~(lv_mask | (seg2d == int(myo)))          # atrium counts as "open" background
    boundary = lv_mask & ~ndimage.binary_erosion(lv_mask)
    open_boundary = boundary & ndimage.binary_dilation(bg)

    rr, cc = np.nonzero(lv_mask)
    if open_boundary.sum() >= 3:
        obr, obc = np.nonzero(open_boundary)
        base_mid = np.array([obr.mean(), obc.mean()], dtype=float)
    else:                                          # fully enclosed LV -> centroid fallback
        base_mid = np.array([rr.mean(), cc.mean()], dtype=float)

    d = np.hypot(rr - base_mid[0], cc - base_mid[1])
    apex = np.array([rr[d.argmax()], cc[d.argmax()]], dtype=float)
    v = apex - base_mid
    length = float(np.linalg.norm(v))
    if length < 1e-3:
        raise RuntimeError("degenerate LV long axis")
    return apex, base_mid, v / length, length


def _signed_distance(mask):
    """Signed distance to the mask boundary: >0 outside, <0 inside, ~0 on the border."""
    mask = mask.astype(bool)
    if not mask.any():
        return np.full(mask.shape, np.inf)
    if mask.all():
        return np.full(mask.shape, -np.inf)
    return ndimage.distance_transform_edt(~mask) - ndimage.distance_transform_edt(mask)


def _displace(mask, offset_field):
    """Move a region's border outward (offset>0) / inward (offset<0) by a smooth field.

    Thresholding the SDF at a spatially-varying offset = morphological dilation/erosion
    by a locally-varying radius, which keeps the border smooth.
    """
    return _signed_distance(mask) < offset_field


def _apex_weight(shape, apex, length, frac=0.35):
    """Smooth [0,1] weight peaking at the apex, decaying over the apical region."""
    sigma = max(frac * length, 6.0)
    rows = np.arange(shape[0])[:, None]
    cols = np.arange(shape[1])[None, :]
    dist2 = (rows - apex[0]) ** 2 + (cols - apex[1]) ** 2
    return np.exp(-dist2 / (2.0 * sigma ** 2))


def _largest_component(mask):
    """Keep only the largest connected component of a boolean mask."""
    if not mask.any():
        return mask
    lab, n = ndimage.label(mask)
    if n <= 1:
        return mask
    keep = np.bincount(lab.ravel())[1:].argmax() + 1
    return lab == keep


def _compose(lv_mask, epi_mask, atrium_mask, lv=Label.LV, myo=Label.MYO):
    """Rebuild a clean label map: LV cavity inside a myocardial ring (epi = LV|MYO)."""
    epi_mask = _largest_component(ndimage.binary_fill_holes(epi_mask))
    lv_mask = _largest_component(lv_mask & epi_mask)
    lv_mask = ndimage.binary_fill_holes(lv_mask)
    out = np.zeros(lv_mask.shape, dtype=np.int16)
    out[epi_mask] = int(myo)
    out[lv_mask] = int(lv)
    out[atrium_mask & (out == 0)] = int(Label.ATRIUM)   # preserve untouched atrium
    return out


def corrupt_apex_frame(seg2d, spacing, mode, mag_mm, lv=Label.LV, myo=Label.MYO):
    """Apply one smooth apex morphological deformation to a 2-D label frame.

    Args:
        seg2d: (H, W) int label map.
        spacing: (row_mm, col_mm) in-plane spacing (``img_nifti.header['pixdim'][1:3]``).
        mode: one of ``MODES`` (must be concrete, not ``"random"`` -- the volume-level
            wrapper resolves randomness once so all frames share the same mode).
        mag_mm: apex border displacement in millimetres.
        lv, myo: label ids.

    Returns:
        (corrupted (H, W) int array, changed: bool). ``changed=False`` => frame left
        unchanged (no LV, apex not found, or a no-op) so callers can skip it.
    """
    seg2d = np.asarray(seg2d)
    lv0 = seg2d == int(lv)
    myo0 = seg2d == int(myo)
    if not lv0.any() or not myo0.any():
        return seg2d.copy(), False

    mean_sp = float(np.mean(np.abs(np.asarray(spacing, dtype=float))))
    amp_px = max(mag_mm / max(mean_sp, 1e-6), 1.0)

    try:
        apex, _base_mid, _u, length = _apex_and_axis(seg2d, lv=lv, myo=myo)
    except (RuntimeError, ValueError, IndexError):
        return seg2d.copy(), False

    epi0 = lv0 | myo0
    atrium = seg2d == int(Label.ATRIUM)
    w = _apex_weight(seg2d.shape, apex, length)

    if mode == "balloon_apex":                     # tip inflates & displaces outward
        epi1 = _displace(epi0, amp_px * w)
        lv1 = _displace(lv0, amp_px * w)
    elif mode == "shrink_apex":                    # rounded foreshortening
        epi1 = _displace(epi0, -amp_px * w)
        lv1 = _displace(lv0, -amp_px * w)
    elif mode == "thin_myo":                       # endocardium moves out -> thin wall
        epi1 = epi0
        lv1 = _displace(lv0, amp_px * w)
    elif mode == "thick_myo":                      # endocardium moves in -> thick wall
        epi1 = epi0
        lv1 = _displace(lv0, -amp_px * w)
    elif mode == "aneurysm":                       # cavity bulges, wall thins
        epi1 = _displace(epi0, 0.35 * amp_px * w)
        lv1 = _displace(lv0, amp_px * w)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    out = _compose(lv1, epi1, atrium, lv=lv, myo=myo)
    return out, bool((out != seg2d).any())


def corrupt_apex_volume(seg, spacing, mode="random", mag_mm=6.0, rng=None,
                        lv=Label.LV, myo=Label.MYO):
    """Apply ONE consistent apex deformation to every frame (last axis) of a volume.

    The mode (if ``"random"``) and magnitude are resolved once here and held fixed across
    all frames, so a corrupted sequence shows a single coherent deformation; only the
    apex position is re-tracked per frame. Accepts ``(H, W, frames)`` or
    ``(1, H, W, frames)`` and returns the SAME shape, plus the number of frames changed.
    """
    rng = rng or np.random.default_rng()
    seg = np.asarray(seg)
    had_channel = seg.ndim == 4
    vol = seg[0] if had_channel else seg           # -> (H, W, frames)

    resolved_mode = rng.choice(MODES) if mode == "random" else mode

    out = vol.copy()
    n_changed = 0
    for i in range(vol.shape[-1]):
        frame, changed = corrupt_apex_frame(vol[..., i], spacing, resolved_mode, mag_mm, lv, myo)
        out[..., i] = frame
        n_changed += int(changed)

    if had_channel:
        out = out[None]
    return out, n_changed
