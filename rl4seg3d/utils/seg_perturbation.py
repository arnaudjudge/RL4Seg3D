"""Synthetic perturbation of the basal edge of an LV/MYO segmentation.

Used to generate richer training data for the landmark (mitral valve commissure)
reward net: it deliberately pushes the segmentation base off the valve plane so the
reward net sees a controlled distribution of base-placement errors.

Only the segmentation label map is modified — the image is never touched.

Label convention (matches ``vital.data.camus.config.Label`` and the root ``mask_utils``):
    BG = 0, LV = 1, MYO = 2, ATRIUM = 3
"""
import numpy as np

_BG, _LV, _MYO, _ATRIUM = 0, 1, 2, 3


def _apex_base_axis(seg, ys, xs):
    """Apex->base axis for an apex-up echo ventricle.

    Returns ``(d, perp, anchor, length, halfwidth)``: ``d`` the unit apex->base axis,
    ``perp`` the commissure axis, ``anchor`` the apex point (held fixed by the stretch),
    ``length`` the structure's vertical extent, and ``halfwidth`` its half-width.

    These are standard apical echo views: the apex is at the top of the image and the
    base (mitral valve) at the bottom, so the apical-basal axis is just vertical, pointing
    down. PCA is *not* used: these ventricle masks are wider than tall, so its major axis
    lands on the commissure (horizontal) direction; and no robust shape cue separates apex
    from base, since the segmentation closes the base with a myo wedge as large as the
    apex cap. (``seg`` is accepted for signature symmetry but the orientation is fixed.)
    """
    top, bot = ys.min(), ys.max()
    d = np.array([1.0, 0.0])           # apex (top) -> base (bottom)
    perp = np.array([0.0, 1.0])        # commissure (left-right) axis
    anchor = np.array([float(top), float(xs.mean())])
    return d, perp, anchor, float(bot - top), float((xs.max() - xs.min()) / 2.0)


def _perturb_frame(seg, shift_px, asymmetry):
    """Stretch a single 2D label map along its long axis. Returns a new array.

    ``shift_px > 0`` lengthens the ventricle so the base (myo cap + cavity together)
    slides ``shift_px`` pixels past the valve; ``shift_px < 0`` shrinks it so the base
    pulls back. The apex is held fixed. ``asymmetry`` in [-1, 1] makes one commissure
    stretch more than the other (wedge-shaped base error). The warp is continuous, so the
    cavity lengthens smoothly and the existing cap is moved rather than a new one stacked.
    """
    struct = (seg == _LV) | (seg == _MYO)
    if not struct.any() or shift_px == 0:
        return seg.copy()

    H, W = seg.shape
    ys, xs = np.nonzero(struct)
    d, perp, anchor, length, halfwidth = _apex_base_axis(seg, ys, xs)
    if length < 1.0:
        return seg.copy()

    # Keep the stretched base inside the frame: pushing it past the image bottom would
    # slice it into a straight edge at the boundary. Cap the (asymmetry-amplified)
    # downward displacement to the room below the current base.
    if shift_px > 0:
        room = (H - 2) - ys.max()
        shift_px = min(float(shift_px), max(0.0, room / (1.0 + abs(asymmetry))))
        if shift_px < 1.0:
            return seg.copy()

    # Stretch each column to its OWN base depth so the base displacement is uniform
    # across the width (both commissures move by ~shift_px) rather than proportional to
    # how deep each column's base sits — otherwise the higher (left) commissure always
    # under-moves and looks unperturbed. `asymmetry` then tilts the displacement to make
    # one commissure move more (a wedge). The apex (row == anchor row) is held fixed.
    anchor_row, anchor_col = anchor
    row_idx = np.where(struct, np.arange(H)[:, None], -1)
    base_off_col = row_idx.max(axis=0).astype(float) - anchor_row  # base depth per column

    cols = np.arange(W)
    bnorm = np.clip((cols - anchor_col) / (halfwidth + 1e-6), -1, 1)
    disp_col = shift_px * (1.0 + asymmetry * bnorm)
    with np.errstate(divide="ignore", invalid="ignore"):
        f_col = (base_off_col + disp_col) / base_off_col
    f_col = np.clip(np.where(base_off_col > 5.0, f_col, 1.0), 0.05, 4.0)  # guard tips

    rr, cc = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    sy = np.round(anchor_row + (rr - anchor_row) / f_col[cc]).astype(int)  # inverse warp
    inb = (sy >= 0) & (sy < H)

    warped = np.zeros((H, W), dtype=seg.dtype)
    warped[inb] = seg[sy[inb], cc[inb]]

    out = np.where(struct, _BG, seg)  # clear old structure, keep any other labels
    ws = (warped == _LV) | (warped == _MYO)
    out[ws] = warped[ws]
    return out


def perturb_segmentation_base(
    seg,
    shift_range_mm=(-15.0, 15.0),
    asymmetry_range=(-0.5, 0.5),
    mm_per_px=0.37,
    temporal="correlated",
    p_apply=0.85,
    rng=None,
):
    """Push the basal edge of an LV/MYO segmentation off the mitral valve plane.

    Synthesizes mitral-valve commissure landmark errors for the reward-net dataset.
    Operates ONLY on the segmentation; the input array is left untouched.

    Args:
        seg: integer label map, shape ``(T, H, W)`` or ``(H, W)``. Labels BG/LV/MYO/
            ATRIUM = 0/1/2/3.
        shift_range_mm: ``(min, max)`` base displacement in millimetres. Positive sits the
            base below the valve, negative above it. Sampled from a triangular
            distribution peaked at 0 mm, so most errors are small (the hard, near-threshold
            regime where the graded reward still carries gradient) with a tail to clearly
            wrong. A span crossing zero produces both directions.
        asymmetry_range: ``(min, max)`` per-commissure tilt in [-1, 1]; a non-zero value
            makes one commissure shift more than the other (wedge error).
        mm_per_px: in-plane pixel spacing of the segmentation grid; the perturbation runs
            at the resampled spacing (``common_spacing`` in predict3d.yaml), so mm are
            converted to pixels with this. Keeps the magnitude resolution-proof.
        temporal: ``"correlated"`` draws one (shift, asymmetry) for the whole sequence
            (mimics a consistently biased policy); ``"independent"`` resamples per frame.
        p_apply: probability of perturbing at all; otherwise an unchanged copy is returned,
            so the net also sees correct, base-on-valve (high-reward) examples.
        rng: ``np.random.Generator`` for reproducibility (created if ``None``).

    Returns:
        Perturbed integer label map, same shape and dtype as the input.
    """
    rng = rng if rng is not None else np.random.default_rng()
    arr = np.asarray(seg)
    squeeze = arr.ndim == 2
    frames = arr[None] if squeeze else arr
    out = frames.copy()

    if rng.random() < p_apply:
        lo, hi = shift_range_mm
        mode = min(max(0.0, lo), hi)  # triangular peak at 0 mm (clamped into the span)

        def _sample():
            return (rng.triangular(lo, mode, hi),
                    float(rng.uniform(*asymmetry_range)))

        shift_mm, asym = _sample()
        shift_px = 0
        for t in range(len(frames)):
            if temporal == "independent":
                shift_mm, asym = _sample()
            shift_px = int(round(shift_mm / mm_per_px))
            out[t] = _perturb_frame(frames[t], shift_px, asym)
        print(f"base perturb: shift={shift_mm:.1f}mm ({shift_px}px) asym={asym:.2f}")

    return out[0] if squeeze else out
