# src/sharp/utils/stereo_align.py
"""
StereoPhotoMaker-style automatic stereo alignment + auto-crop.

This module provides a robust, fully-automated pipeline intended for typical
left/right stereo pairs (including SBS splits). The design goals are:

- Prioritize stereo comfort: minimize vertical disparity aggressively.
- Avoid "explaining away" true stereo parallax with overly-flexible warps.
- Use homography only when it clearly helps and does not severely distort.
- Auto-crop to the maximal overlapping valid region after warping.

Public API:
    estimate_transform(L, R) -> (model, M, stats)
    warp_and_crop(L, R, M, model) -> (L_crop, R_aligned_crop, crop_rect)
    auto_align_and_crop(L, R) -> (L_crop, R_aligned_crop, meta)

Inputs:
    L, R: numpy arrays shaped (H, W) or (H, W, C). Color arrays should be uint8.

Dependencies:
    OpenCV (cv2) is required.

Notes:
- The primary "SPM-like" behavior comes from:
    1) model ladder + gating (prefer affine-partial)
    2) final median vertical correction based on inlier correspondences
    3) overlap-based auto-cropping

- This implementation warps R into L's coordinate system (common & stable).
  You can add symmetric warping later (apply half transforms) for affine models.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np

try:
    import cv2
except Exception as e:  # pragma: no cover
    raise ImportError(
        "stereo_align.py requires OpenCV (cv2). Install opencv-python or opencv-python-headless."
    ) from e


ModelName = Literal["affine", "homography"]


@dataclass
class AlignParams:
    # Feature detection / matching
    max_features: int = 8000
    ratio_test: float = 0.75  # Lowe ratio
    # Estimation scale for speed/stability
    work_width: int = 1600
    # RANSAC thresholds are in pixels at working scale
    ransac_thresh: float = 3.0
    min_good_matches: int = 40
    min_inliers: int = 25

    # Model selection / gating
    prefer_affine: bool = True
    homography_err_gain: float = 0.80  # choose H only if errH < gain*errA
    homography_max_corner_warp: float = 0.04  # fraction of width; limits distortion

    # Refinement / enforcement
    force_vertical_correction: bool = True

    # Cropping
    crop_inset_px: int = 4  # inset after overlap crop to avoid edge artifacts


def _to_gray_u8(img: np.ndarray) -> np.ndarray:
    """Convert (H,W[,C]) to grayscale uint8 for feature extraction."""
    if img.ndim == 2:
        g = img
    elif img.ndim == 3:
        # handle RGB/BGR indistinguishably for gray conversion
        if img.shape[2] == 3:
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            # take first channel if odd shape
            g = img[:, :, 0]
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    if g.dtype != np.uint8:
        # normalize to 0..255
        g = np.clip(g, 0, 255).astype(np.uint8)
    return g


def _resize_keep_aspect(img: np.ndarray, target_w: int) -> Tuple[np.ndarray, float]:
    """Resize to target width, keep aspect. Returns resized image and scale factor (resized/original)."""
    h, w = img.shape[:2]
    if w <= target_w:
        return img, 1.0
    scale = target_w / float(w)
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (target_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def _detect_and_match(
    grayL: np.ndarray,
    grayR: np.ndarray,
    max_features: int,
    ratio_test: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Detect keypoints & descriptors and return matched point arrays (ptsR -> ptsL).

    Returns:
        ptsR: (N,1,2) float32
        ptsL: (N,1,2) float32
        info: dict about detector and match counts
    """
    info: Dict[str, Any] = {}

    # Prefer SIFT if available (best general-purpose); else AKAZE; else ORB.
    detector = None
    norm_type = None

    if hasattr(cv2, "SIFT_create"):
        detector = cv2.SIFT_create(nfeatures=max_features)
        norm_type = cv2.NORM_L2
        info["detector"] = "SIFT"
    else:
        # AKAZE is generally better than ORB on many natural images
        try:
            detector = cv2.AKAZE_create()
            norm_type = cv2.NORM_HAMMING
            info["detector"] = "AKAZE"
        except Exception:
            detector = cv2.ORB_create(nfeatures=max_features)
            norm_type = cv2.NORM_HAMMING
            info["detector"] = "ORB"

    kL, dL = detector.detectAndCompute(grayL, None)
    kR, dR = detector.detectAndCompute(grayR, None)
    info["kpts_L"] = 0 if kL is None else len(kL)
    info["kpts_R"] = 0 if kR is None else len(kR)

    if dL is None or dR is None or len(kL) < 8 or len(kR) < 8:
        return (
            np.zeros((0, 1, 2), np.float32),
            np.zeros((0, 1, 2), np.float32),
            info,
        )

    matcher = cv2.BFMatcher(normType=norm_type, crossCheck=False)
    raw = matcher.knnMatch(dR, dL, k=2)  # map R -> L
    info["raw_matches"] = len(raw)

    good = []
    for pair in raw:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio_test * n.distance:
            good.append(m)
    info["good_matches"] = len(good)

    if len(good) == 0:
        return (
            np.zeros((0, 1, 2), np.float32),
            np.zeros((0, 1, 2), np.float32),
            info,
        )

    ptsR = np.float32([kR[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    ptsL = np.float32([kL[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    return ptsR, ptsL, info


def _reprojection_error(
    M: np.ndarray,
    ptsR: np.ndarray,
    ptsL: np.ndarray,
    model: ModelName,
) -> float:
    """Median reprojection error in pixels."""
    if ptsR.shape[0] == 0:
        return float("inf")
    if model == "homography":
        pred = cv2.perspectiveTransform(ptsR, M)
    else:
        pred = cv2.transform(ptsR, M)
    dif = pred - ptsL
    err = np.sqrt((dif[:, 0, 0] ** 2) + (dif[:, 0, 1] ** 2))
    return float(np.median(err))


def _corner_warp_ratio_homography(H: np.ndarray, w: int, h: int) -> float:
    """
    Measure how severe the homography is by how far image corners move,
    normalized by width. Used to gate against over-warping (stereo safety).
    """
    corners = np.float32(
        [[[0, 0]], [[w - 1, 0]], [[w - 1, h - 1]], [[0, h - 1]]]
    )  # (4,1,2)
    warped = cv2.perspectiveTransform(corners, H)
    disp = warped - corners
    d = np.sqrt((disp[:, 0, 0] ** 2) + (disp[:, 0, 1] ** 2))
    return float(np.max(d) / max(1.0, float(w)))


def estimate_transform(
    L: np.ndarray,
    R: np.ndarray,
    params: Optional[AlignParams] = None,
) -> Tuple[ModelName, np.ndarray, Dict[str, Any]]:
    """
    Estimate a stereo-safe transform mapping R -> L coordinates.

    Returns:
        model: "affine" or "homography"
        M:     2x3 affine matrix if model=="affine", else 3x3 homography
        stats: diagnostics including errors, inliers, etc.

    Raises:
        RuntimeError if not enough matches/inliers to estimate any model.
    """
    if params is None:
        params = AlignParams()

    stats: Dict[str, Any] = {}

    grayL = _to_gray_u8(L)
    grayR = _to_gray_u8(R)

    grayLw, sL = _resize_keep_aspect(grayL, params.work_width)
    grayRw, sR = _resize_keep_aspect(grayR, params.work_width)

    # For stereo pairs, L and R widths should match; if not, we still proceed.
    # We'll estimate on their working-res versions separately.
    ptsR, ptsL, minfo = _detect_and_match(
        grayLw, grayRw, params.max_features, params.ratio_test
    )
    stats.update(minfo)
    stats["work_scale_L"] = sL
    stats["work_scale_R"] = sR

    if ptsR.shape[0] < params.min_good_matches:
        raise RuntimeError(
            f"Not enough good matches to estimate transform: {ptsR.shape[0]} < {params.min_good_matches}"
        )

    # Estimate affine-partial (similarity-ish)
    A, inA = cv2.estimateAffinePartial2D(
        ptsR, ptsL, method=cv2.RANSAC, ransacReprojThreshold=params.ransac_thresh
    )
    inliersA = 0 if inA is None else int(np.sum(inA))
    stats["affine_inliers"] = inliersA

    # Estimate homography
    H, inH = cv2.findHomography(
        ptsR, ptsL, method=cv2.RANSAC, ransacReprojThreshold=params.ransac_thresh
    )
    inliersH = 0 if inH is None else int(np.sum(inH))
    stats["homography_inliers"] = inliersH

    have_affine = A is not None and inliersA >= params.min_inliers
    have_h = H is not None and inliersH >= params.min_inliers

    if not have_affine and not have_h:
        raise RuntimeError(
            f"Failed to estimate transform. Inliers: affine={inliersA}, homography={inliersH}"
        )

    # Compute errors using inliers only when available
    def _filter_inliers(pts: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
        if mask is None:
            return pts
        m = mask.astype(bool).reshape(-1)
        return pts[m]

    errA = float("inf")
    errH = float("inf")

    if have_affine:
        ptsR_A = _filter_inliers(ptsR, inA)
        ptsL_A = _filter_inliers(ptsL, inA)
        errA = _reprojection_error(A, ptsR_A, ptsL_A, "affine")

    corner_ratio = float("inf")
    if have_h:
        ptsR_H = _filter_inliers(ptsR, inH)
        ptsL_H = _filter_inliers(ptsL, inH)
        errH = _reprojection_error(H, ptsR_H, ptsL_H, "homography")
        # severity computed at working-res dimensions
        wW = grayLw.shape[1]
        hW = grayLw.shape[0]
        corner_ratio = _corner_warp_ratio_homography(H, wW, hW)

    stats["affine_err_med_px"] = errA
    stats["homography_err_med_px"] = errH
    stats["homography_corner_warp_ratio"] = corner_ratio

    # Model selection: prefer affine unless homography is clearly better and not too distortive
    choose: ModelName = "affine" if have_affine else "homography"
    if have_h:
        if not have_affine:
            choose = "homography"
        else:
            homography_good = (
                (errH < params.homography_err_gain * errA)
                and (corner_ratio <= params.homography_max_corner_warp)
            )
            if not params.prefer_affine:
                # if not preferring affine, allow H whenever it is at least slightly better and safe
                homography_good = homography_good or (
                    (errH < errA) and (corner_ratio <= params.homography_max_corner_warp)
                )
            if homography_good:
                choose = "homography"

    stats["chosen_model"] = choose

    # Rescale transform back to full-res if we estimated on scaled images.
    # We estimated mapping in working pixel coordinates. If sL != sR, we handle separately.
    # Let full coords be x_full, working coords x_work = s * x_full.
    # We have: xL_work = M_work * xR_work
    # Substitute: sL * xL_full = M_work * (sR * xR_full)
    # => xL_full = (1/sL) * M_work * (sR) * xR_full
    # For affine: 2x3 with last col translation.
    def _rescale_affine(M: np.ndarray, sL_: float, sR_: float) -> np.ndarray:
        M2 = M.copy().astype(np.float64)
        # linear part scales by (sR/sL)
        M2[0:2, 0:2] *= (sR_ / sL_)
        # translation scales by (1/sL)
        M2[0:2, 2] *= (1.0 / sL_)
        return M2.astype(np.float32)

    def _rescale_homography(Hm: np.ndarray, sL_: float, sR_: float) -> np.ndarray:
        # xL_full = S_L^-1 * H_work * S_R * xR_full
        SL_inv = np.array([[1.0 / sL_, 0, 0], [0, 1.0 / sL_, 0], [0, 0, 1]], np.float64)
        SR = np.array([[sR_, 0, 0], [0, sR_, 0], [0, 0, 1]], np.float64)
        Hf = SL_inv @ Hm.astype(np.float64) @ SR
        return Hf.astype(np.float32)

    if choose == "affine":
        M_full = _rescale_affine(A, sL, sR)
    else:
        M_full = _rescale_homography(H, sL, sR)

    return choose, M_full, stats


def _warp_image(
    img: np.ndarray,
    M: np.ndarray,
    model: ModelName,
    out_w: int,
    out_h: int,
    interp: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    if model == "homography":
        return cv2.warpPerspective(img, M, (out_w, out_h), flags=interp)
    return cv2.warpAffine(img, M, (out_w, out_h), flags=interp)


def _warp_mask(
    h: int,
    w: int,
    M: np.ndarray,
    model: ModelName,
    out_w: int,
    out_h: int,
) -> np.ndarray:
    mask = np.ones((h, w), np.uint8) * 255
    if model == "homography":
        return cv2.warpPerspective(mask, M, (out_w, out_h), flags=cv2.INTER_NEAREST)
    return cv2.warpAffine(mask, M, (out_w, out_h), flags=cv2.INTER_NEAREST)


def _apply_vertical_median_correction(
    M: np.ndarray,
    model: ModelName,
    ptsR: np.ndarray,
    ptsL: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Compute median vertical residual after transform and apply a corrective y-translation.

    Returns:
        M2: corrected matrix
        dy: median residual in pixels (positive means warpedR is below L)
    """
    if ptsR.shape[0] == 0:
        return M, 0.0

    if model == "homography":
        pred = cv2.perspectiveTransform(ptsR, M)
        dy = float(np.median(pred[:, 0, 1] - ptsL[:, 0, 1]))
        M2 = M.copy().astype(np.float64)
        # apply translation in output y: y' = y - dy
        T = np.array([[1, 0, 0], [0, 1, -dy], [0, 0, 1]], np.float64)
        M2 = T @ M2
        return M2.astype(np.float32), dy
    else:
        pred = cv2.transform(ptsR, M)
        dy = float(np.median(pred[:, 0, 1] - ptsL[:, 0, 1]))
        M2 = M.copy().astype(np.float64)
        M2[1, 2] -= dy
        return M2.astype(np.float32), dy


def warp_and_crop(
    L: np.ndarray,
    R: np.ndarray,
    M: np.ndarray,
    model: ModelName,
    params: Optional[AlignParams] = None,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """
    Warp R into L space and crop both to maximal overlapping valid region.

    Returns:
        L_crop: (h2, w2[,c])
        R_crop: (h2, w2[,c])  (aligned/warped)
        crop_rect: (x0, y0, x1, y1) inclusive coordinates in L space before cropping
    """
    if params is None:
        params = AlignParams()

    hL, wL = L.shape[:2]
    out_w, out_h = wL, hL

    Rw = _warp_image(R, M, model, out_w=out_w, out_h=out_h, interp=cv2.INTER_LINEAR)

    maskL = np.ones((out_h, out_w), np.uint8) * 255
    maskR = _warp_mask(R.shape[0], R.shape[1], M, model, out_w=out_w, out_h=out_h)

    overlap = cv2.bitwise_and(maskL, maskR)
    ys, xs = np.where(overlap > 0)
    if xs.size == 0 or ys.size == 0:
        raise RuntimeError("No overlap after warping; transform may be invalid.")

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())

    inset = int(max(0, params.crop_inset_px))
    x0 = min(max(x0 + inset, 0), out_w - 1)
    y0 = min(max(y0 + inset, 0), out_h - 1)
    x1 = min(max(x1 - inset, x0), out_w - 1)
    y1 = min(max(y1 - inset, y0), out_h - 1)

    Lc = L[y0 : y1 + 1, x0 : x1 + 1]
    Rc = Rw[y0 : y1 + 1, x0 : x1 + 1]
    return Lc, Rc, (x0, y0, x1, y1)


def auto_align_and_crop(
    L: np.ndarray,
    R: np.ndarray,
    params: Optional[AlignParams] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Fully automated pipeline: estimate transform, enforce vertical correction, warp, crop.

    Returns:
        L_crop, R_crop, meta

    meta includes:
        - chosen_model
        - matrix (final)
        - crop_rect
        - stats (feature/match/fit diagnostics)
        - vertical_correction_dy
    """
    if params is None:
        params = AlignParams()

    model, M, stats = estimate_transform(L, R, params=params)

    # Recompute correspondences for vertical correction at working scale for stability.
    # This uses the same detection pipeline as estimation, but on working-res images.
    grayL = _to_gray_u8(L)
    grayR = _to_gray_u8(R)
    grayLw, sL = _resize_keep_aspect(grayL, params.work_width)
    grayRw, sR = _resize_keep_aspect(grayR, params.work_width)
    ptsR, ptsL, _ = _detect_and_match(grayLw, grayRw, params.max_features, params.ratio_test)

    # Convert working points to full-res coordinates
    # pts_work = s * pts_full  => pts_full = pts_work / s
    if ptsR.shape[0] > 0:
        ptsR_full = ptsR.copy()
        ptsL_full = ptsL.copy()
        ptsR_full[:, 0, :] /= max(1e-12, float(sR))
        ptsL_full[:, 0, :] /= max(1e-12, float(sL))
    else:
        ptsR_full = ptsR
        ptsL_full = ptsL

    dy = 0.0
    M_final = M
    if params.force_vertical_correction and ptsR_full.shape[0] >= params.min_inliers:
        M_final, dy = _apply_vertical_median_correction(M, model, ptsR_full, ptsL_full)

    Lc, Rc, crop_rect = warp_and_crop(L, R, M_final, model, params=params)

    meta: Dict[str, Any] = {
        "chosen_model": model,
        "matrix": M_final,
        "crop_rect": crop_rect,
        "stats": stats,
        "vertical_correction_dy": dy,
    }
    return Lc, Rc, meta
