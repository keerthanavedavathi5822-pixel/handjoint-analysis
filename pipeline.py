# -*- coding: utf-8 -*-
"""
pipeline.py — Standalone Multimodal RGB + FLIR Radiometric Hand Joint Thermal Pipeline.

Cleaned for Streamlit web deployment from the Colab research pipeline:
  - MediaPipe 21-point RGB hand landmark extraction (Tasks API + legacy fallback)
  - Radiometric .npy (single stacked or per-frame) & thermal video decoding
  - Automatic RGB-to-thermal similarity calibration (or reuse of rig_calibration.json)
  - Frame alignment, clock-offset estimation, and landmark interpolation
  - Per-joint circular ROI temperature extraction, quality flags, and rolling temporal features
  - Annotated RGB + Thermal overlay image generation for live display in Streamlit
"""

import math
import re
import tempfile
import urllib.request
import warnings
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.optimize import minimize
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Constants & Default Configuration
# -----------------------------------------------------------------------------
DEFAULT_CFG = {
    # Acquisition
    "RGB_FPS": 24.0,
    "THERMAL_FPS": 9.0,
    "THERMAL_TIMEBASE": "auto",  # auto | aligned_to_rgb | fixed_fps
    # Thermal decoding
    "RADIOMETRIC_UNIT": "auto",  # auto | celsius | kelvin | decikelvin | centikelvin | k004
    "THERMAL_KIND": "auto",      # auto | gray | colormap | radiometric_centikelvin | celsius | gray16
    "THERMAL_COLORMAP": "auto",
    "TEMP_MIN_C": 20.0,
    "TEMP_MAX_C": 40.0,
    "MAX_THERMAL_FRAMES": None,
    # MediaPipe
    "MP_DET_CONF": 0.5,
    "MP_TRACK_CONF": 0.5,
    "RGB_MAX_SIDE": 960,
    # ROI & features
    "ROI_ALPHA": 0.10,
    "ROI_ALPHA_TIP": 0.08,
    "MIN_ROI_PX": 2.0,
    "MIN_ON_SKIN": 0.70,
    "WINDOW_S": 1.0,
    # Quality thresholds
    "LOW_CONF": 0.6,
    "SKEW_TOL_MS": 25.0,
    "FAST_MOTION_REL": 0.15,
    "LOW_CONTRAST_C": 1.5,
    "TEMP_PLAUSIBLE": (15.0, 45.0),
    "CALIB_OUTSIDE_FRAC": 0.4,
    "TIP_OVERLAP_REL": 0.15,
    # Calibration & overlays
    "CALIB_FRAMES": 12,
    "OVERLAY_FRAMES": 6,
}

JOINTS = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]

FINGERS = {
    "THUMB": [1, 2, 3, 4],
    "INDEX": [5, 6, 7, 8],
    "MIDDLE": [9, 10, 11, 12],
    "RING": [13, 14, 15, 16],
    "PINKY": [17, 18, 19, 20],
}

TIPS = [4, 8, 12, 16, 20]

BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

CRITICAL = {
    "NO_HAND", "LOW_CONF", "OUT_OF_FRAME",
    "ROI_OFF_SKIN", "TEMP_OUT_OF_RANGE", "CALIB_ERROR",
}

CMAPS = {
    "INFERNO": cv2.COLORMAP_INFERNO,
    "JET": cv2.COLORMAP_JET,
    "HOT": cv2.COLORMAP_HOT,
    "PLASMA": cv2.COLORMAP_PLASMA,
    "MAGMA": cv2.COLORMAP_MAGMA,
    "TURBO": cv2.COLORMAP_TURBO,
    "RAINBOW": cv2.COLORMAP_RAINBOW,
    "VIRIDIS": cv2.COLORMAP_VIRIDIS,
    "BONE": cv2.COLORMAP_BONE,
}

UNITS = {
    "celsius": (1.0, 0.0),
    "kelvin": (1.0, -273.15),
    "decikelvin": (0.1, -273.15),
    "centikelvin": (0.01, -273.15),
    "k004": (0.04, -273.15),
}

HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


# -----------------------------------------------------------------------------
# Section 3: Video Readers & Thermal -> Temperature Decoding
# -----------------------------------------------------------------------------
def iter_frames(path):
    cap = cv2.VideoCapture(str(path))
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        yield i, fr
        i += 1
    cap.release()


def video_meta(path):
    cap = cv2.VideoCapture(str(path))
    meta = dict(
        n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        fps=float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    cap.release()
    return meta


class ThermalDecoder:
    """Turns a raw thermal video frame (gray / colour-mapped / radiometric) into float32 deg C."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.kind = cfg["THERMAL_KIND"]
        self.cmap = cfg["THERMAL_COLORMAP"]
        self.fitted = False
        self.table = None
        self.absolute = False

    def _lut(self, name):
        return (
            cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(-1, 1), CMAPS[name])
            .reshape(256, 3)
            .astype(np.float32)
        )

    def _build_table(self, name):
        g = (np.arange(32) * 8 + 4).astype(np.float32)
        B, G, R = np.meshgrid(g, g, g, indexing="ij")
        _, idx = cKDTree(self._lut(name)).query(np.stack([B.ravel(), G.ravel(), R.ravel()], 1))
        return idx.reshape(32, 32, 32).astype(np.uint8)

    def fit(self, fr):
        if fr.ndim == 3 and fr.shape[2] == 4:
            fr = fr[..., :3]
        if self.kind == "auto":
            if fr.dtype != np.uint8:
                med = float(np.median(fr))
                self.kind = (
                    "radiometric_centikelvin" if med > 20000
                    else ("celsius" if fr.dtype.kind == "f" and 5 < med < 80 else "gray16")
                )
            elif (
                fr.ndim == 2
                or np.abs(fr[..., 0].astype(int) - fr[..., 1]).mean()
                + np.abs(fr[..., 1].astype(int) - fr[..., 2]).mean()
                < 6
            ):
                self.kind = "gray"
            else:
                self.kind = "colormap"

        if self.kind == "colormap":
            if self.cmap == "auto":
                px = fr.reshape(-1, 3)[:: max(1, fr.shape[0] * fr.shape[1] // 6000)].astype(np.float32)
                best = None
                for name in CMAPS:
                    d = float(cKDTree(self._lut(name)).query(px)[0].mean())
                    if best is None or d < best[1]:
                        best = (name, d)
                self.cmap = best[0]
                if best[1] > 25:
                    self.kind = "gray"
            if self.kind == "colormap":
                self.table = self._build_table(self.cmap)

        self.absolute = self.kind in ("colormap", "radiometric_centikelvin", "celsius")
        self.fitted = True

    def __call__(self, fr):
        lo, hi = self.cfg["TEMP_MIN_C"], self.cfg["TEMP_MAX_C"]
        if fr.ndim == 3 and fr.shape[2] == 4:
            fr = fr[..., :3]
        if self.kind == "radiometric_centikelvin":
            return fr.astype(np.float32) * 0.01 - 273.15
        if self.kind == "celsius":
            return fr.astype(np.float32)
        if self.kind == "gray16":
            f = fr.astype(np.float32)
            a, b = np.percentile(f, [1, 99.5])
            return lo + (f - a) / max(b - a, 1.0) * (hi - lo)
        if self.kind == "gray":
            g = fr if fr.ndim == 2 else fr.mean(-1)
            return lo + g.astype(np.float32) / 255.0 * (hi - lo)
        q = fr >> 3
        idx = self.table[q[..., 0], q[..., 1], q[..., 2]]
        return lo + idx.astype(np.float32) / 255.0 * (hi - lo)


def load_radiometric(rad_paths, cfg, n_hint=None):
    """Load single stacked .npy OR multiple per-frame .npy files into float16 deg C (N, H, W)."""
    if isinstance(rad_paths, (str, Path)):
        rad_paths = [str(rad_paths)]
    rad_paths = sorted([str(p) for p in rad_paths], key=natural_key)

    if len(rad_paths) == 1:
        a = np.load(rad_paths[0], mmap_mode="r")
        if a.ndim == 4:
            a = a[..., 0] if a.shape[-1] in (1, 3) else a[:, 0]
        if a.ndim == 2:
            a = a[None]
        if a.ndim != 3:
            raise ValueError(f"Unexpected radiometric array shape {a.shape} in {rad_paths[0]}")
        ax = 0
        if n_hint and abs(a.shape[0] - n_hint) > 2:
            for cand in (2, 1):
                if abs(a.shape[cand] - n_hint) <= 2:
                    ax = cand
                    break
        if not n_hint and min(a.shape[1], a.shape[2]) < 32 <= a.shape[0]:
            ax = int(np.argmin(a.shape))
        a = np.moveaxis(a, ax, 0)
    else:
        frames = []
        for p in rad_paths:
            arr = np.load(p)
            if arr.ndim == 3:
                arr = arr[..., 0] if arr.shape[-1] in (1, 3) else arr[0]
            frames.append(arr)
        a = np.stack(frames, axis=0)

    n = a.shape[0] if not cfg["MAX_THERMAL_FRAMES"] else min(a.shape[0], cfg["MAX_THERMAL_FRAMES"])
    sample_idx = np.linspace(0, a.shape[0] - 1, min(8, a.shape[0])).astype(int)
    sample = np.asarray(a[sample_idx], dtype=np.float64)
    sample = sample[np.isfinite(sample)]
    med = float(np.median(sample[sample != 0])) if (sample != 0).any() else 0.0

    unit = cfg["RADIOMETRIC_UNIT"]
    if unit == "auto":
        unit = (
            "celsius" if 10 < med < 60
            else "kelvin" if 240 < med < 340
            else "decikelvin" if 2400 < med < 3400
            else "centikelvin" if 24000 < med < 34000
            else "k004" if 6000 < med < 8500
            else "unknown"
        )

    gain, off = UNITS.get(unit, (None, None))
    out = np.empty((n,) + a.shape[1:], np.float16)
    for i in range(0, n, 64):
        c = np.asarray(a[i : i + 64], dtype=np.float32)
        if gain is None:
            lo, hi = np.percentile(c, [1, 99.5])
            c = cfg["TEMP_MIN_C"] + (c - lo) / max(hi - lo, 1e-6) * (cfg["TEMP_MAX_C"] - cfg["TEMP_MIN_C"])
        else:
            c = c * gain + off
        bad = ~np.isfinite(c) | (c < -50) | (c > 150)
        if bad.any():
            for j in range(len(c)):
                if bad[j].any():
                    c[j][bad[j]] = np.median(c[j][~bad[j]]) if (~bad[j]).any() else 25.0
        out[i : i + 64] = c

    info = dict(unit=unit, absolute=gain is not None)
    return out, info


def load_thermal(rad_paths, thermal_video_path, cfg):
    if rad_paths:
        n_hint = video_meta(thermal_video_path)["n"] if thermal_video_path else None
        return load_radiometric(rad_paths, cfg, n_hint=n_hint)

    dec = ThermalDecoder(cfg)
    Ts = []
    for i, fr in iter_frames(thermal_video_path):
        if not dec.fitted:
            dec.fit(fr)
        Ts.append(dec(fr).astype(np.float16))
        if cfg["MAX_THERMAL_FRAMES"] and len(Ts) >= cfg["MAX_THERMAL_FRAMES"]:
            break
    info = dict(unit=f"video:{dec.kind}", absolute=dec.absolute)
    return np.stack(Ts), info


# -----------------------------------------------------------------------------
# Section 4: MediaPipe Landmarks, Hand Mask, Sync & RGB->Thermal Mapping
# -----------------------------------------------------------------------------
def get_hand_model():
    cache_dir = Path(tempfile.gettempdir()) / "thermohand_models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "hand_landmarker.task"
    if not path.exists():
        urllib.request.urlretrieve(HAND_MODEL_URL, str(path))
    return str(path)


def make_hand_detector(cfg):
    """Returns (detect, close) using MediaPipe Tasks API or legacy mp.solutions fallback."""
    if hasattr(mp, "solutions"):
        hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=cfg["MP_DET_CONF"],
            min_tracking_confidence=cfg["MP_TRACK_CONF"],
        )

        def detect(rgb, t_ms):
            res = hands.process(rgb)
            if res.multi_hand_landmarks:
                pts = np.array(
                    [[q.x, q.y, q.z] for q in res.multi_hand_landmarks[0].landmark],
                    np.float32,
                )
                return pts, float(res.multi_handedness[0].classification[0].score)
            return None, 0.0

        return detect, hands.close

    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=get_hand_model()),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=cfg["MP_DET_CONF"],
        min_hand_presence_confidence=cfg["MP_DET_CONF"],
        min_tracking_confidence=cfg["MP_TRACK_CONF"],
    )
    lmk = vision.HandLandmarker.create_from_options(opts)

    def detect(rgb, t_ms):
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = lmk.detect_for_video(img, int(t_ms))
        if res.hand_landmarks:
            pts = np.array([[q.x, q.y, q.z] for q in res.hand_landmarks[0]], np.float32)
            score = float(res.handedness[0][0].score) if res.handedness else 1.0
            return pts, score
        return None, 0.0

    return detect, lmk.close


def extract_rgb_landmarks(rgb_path, cfg):
    detect, close = make_hand_detector(cfg)
    lms, confs, size = [], [], None
    try:
        for i, fr in iter_frames(rgb_path):
            if fr.ndim == 2:
                fr = cv2.cvtColor(fr, cv2.COLOR_GRAY2BGR)
            if fr.dtype != np.uint8:
                fr = cv2.normalize(fr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            h, w = fr.shape[:2]
            size = (w, h)
            s = cfg["RGB_MAX_SIDE"] / max(h, w)
            if s < 1:
                fr = cv2.resize(fr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            t_ms = round(i * 1000.0 / cfg["RGB_FPS"])
            pts, score = detect(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), t_ms)
            if pts is not None:
                lms.append(pts)
                confs.append(score)
            else:
                lms.append(np.full((21, 3), np.nan, np.float32))
                confs.append(0.0)
    finally:
        close()

    if not lms:
        raise ValueError("Could not read any frames from the uploaded RGB video.")
    return dict(
        lm=np.stack(lms),
        conf=np.array(confs, np.float32),
        size=np.array(size),
    )


def align_thermal(T, n_rgb, cfg):
    n = len(T)
    mode = cfg["THERMAL_TIMEBASE"]
    fr, ft = cfg["RGB_FPS"], cfg["THERMAL_FPS"]
    sm = np.stack([T[k][::4, ::4].astype(np.float32) for k in range(n)])
    d = np.abs(np.diff(sm, axis=0)).mean(axis=(1, 2)) if n > 1 else np.zeros(0)
    dup = np.r_[False, d < 0.003]
    dup_frac = float(dup.mean())
    aligned = abs(n / max(n_rgb, 1) - 1) <= 0.05
    use_aligned = aligned if mode == "auto" else (mode == "aligned_to_rgb")

    if use_aligned:
        idx = np.where(~dup)[0] if dup_frac > 0.3 else np.arange(n)
        t = idx / fr
        eff = len(idx) / max(n / fr, 1e-6)
        note = (
            f"{n} thermal vs {n_rgb} RGB frames (aligned; {dup_frac:.0%} repeats removed -> {len(idx)} samples ~{eff:.1f} fps)"
            if dup_frac > 0.3
            else f"{n} thermal vs {n_rgb} RGB frames (aligned at {fr:g} fps)"
        )
    else:
        idx = np.arange(n)
        t = idx / ft
        note = f"{n} thermal frames at fixed {ft:g} fps ({n / ft:.1f}s) vs RGB {n_rgb / fr:.1f}s"

    return (
        T[idx],
        t,
        idx,
        dict(n_thermal_raw=n, n_rgb=n_rgb, dup_frac=dup_frac, aligned=bool(use_aligned), n_kept=int(len(idx)), note=note),
    )


def hand_mask(T):
    Tf = cv2.GaussianBlur(T.astype(np.float32), (0, 0), 1.0)
    lo, hi = np.percentile(Tf, [1, 99.5])
    if hi - lo < 0.3:
        return np.zeros(T.shape, bool), 0.0
    g = np.clip((Tf - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    thr, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = g > thr
    m = ndi.binary_closing(ndi.binary_opening(m, iterations=1), iterations=2)
    m = ndi.binary_fill_holes(m)
    lab, n = ndi.label(m)
    if n == 0:
        return m, 0.0
    m = lab == (1 + int(np.argmax(ndi.sum(m, lab, range(1, n + 1)))))
    contrast = float(np.median(T[m]) - np.median(T[~m])) if (~m).any() else 0.0
    return m, contrast


def hull_mask_px(pts, shape, dilate=0):
    m = np.zeros(shape, np.uint8)
    pts = np.asarray(pts, np.float32)
    if not np.isfinite(pts).all():
        return m.astype(bool)
    cv2.fillConvexPoly(m, cv2.convexHull(pts.reshape(-1, 1, 2)).astype(np.int32), 1)
    if dilate >= 1:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(dilate) + 1,) * 2))
    return m.astype(bool)


def make_A(p, rgb_size, flip):
    s, th, tx, ty = p
    W, H = rgb_size
    c, sn = math.cos(th), math.sin(th)
    R = s * np.array([[c, -sn], [sn, c]])
    Fm = np.array([[-1.0, 0], [0, 1]]) if flip else np.eye(2)
    M = R @ Fm
    return np.hstack([M, (np.array([tx, ty]) - M @ np.array([W / 2, H / 2]))[:, None]])


def warp_pts(xy_norm, rgb_size, A):
    px = xy_norm * np.array(rgb_size, np.float32)
    return px @ A[:, :2].T + A[:, 2]


def fit_similarity(samples, rgb_size, th_shape, flip):
    Ht, Wt = th_shape
    s0 = np.median(
        [
            np.sqrt(m.sum() / max(cv2.contourArea(cv2.convexHull(l.astype(np.float32).reshape(-1, 1, 2))), 1.0))
            for l, m in samples
        ]
    )
    cth = np.mean([np.array(np.nonzero(m)).mean(1)[::-1] for _, m in samples], 0)
    crgb = np.mean([l.mean(0) for l, _ in samples], 0) - np.array(rgb_size) / 2
    if flip:
        crgb[0] = -crgb[0]
    t0 = cth - s0 * crgb
    L0 = s0 * float(np.mean([np.linalg.norm(l[0] - l[9]) for l, _ in samples]))

    def iou_for(p):
        A = make_A(p, rgb_size, flip)
        ious = []
        for l, m in samples:
            w = l @ A[:, :2].T + A[:, 2]
            L = np.linalg.norm(w[0] - w[9]) + 1e-6
            h = hull_mask_px(w, (Ht, Wt), dilate=0.07 * L)
            mm = m & hull_mask_px(w, (Ht, Wt), dilate=0.6 * L0)
            ious.append((h & mm).sum() / max((h | mm).sum(), 1))
        return float(np.mean(ious))

    res = minimize(
        lambda p: -iou_for(p),
        [s0, 0.0, t0[0], t0[1]],
        method="Nelder-Mead",
        options=dict(maxiter=500, xatol=1e-3, fatol=1e-4),
    )
    return res.x, -res.fun


def calibrate_single(rgb, T, t_arr, cfg):
    """Estimate RGB->thermal similarity transform from a single uploaded recording."""
    rgb_size = tuple(int(v) for v in rgb["size"])
    th_shape = T.shape[1:]
    n_frames = cfg["CALIB_FRAMES"]
    samples = []

    for conf_thr in (0.8, 0.5, 0.3):
        samples = []
        idxs = np.linspace(0, len(T) - 1, min(len(T), n_frames * 4)).astype(int)
        for i in idxs:
            j = min(int(round(t_arr[i] * cfg["RGB_FPS"])), len(rgb["lm"]) - 1)
            lm = rgb["lm"][j]
            if not np.isfinite(lm).all() or rgb["conf"][j] < conf_thr:
                continue
            m, c = hand_mask(T[i].astype(np.float32))
            if not (0.02 < m.mean() < 0.75) or c < cfg["LOW_CONTRAST_C"]:
                continue
            samples.append((lm[:, :2] * np.array(rgb_size), m))
            if len(samples) >= n_frames:
                break
        if len(samples) >= 3:
            break

    if len(samples) < 3:
        raise ValueError(
            "Calibration could not find enough usable hand frames. "
            "Try lowering MediaPipe confidence or uploading a saved rig_calibration.json."
        )

    best = None
    for flip in (False, True):
        p, iou = fit_similarity(samples, rgb_size, th_shape, flip)
        if best is None or iou > best[2]:
            best = (p, flip, iou)
    p, flip, iou = best
    return dict(
        params=[float(v) for v in p],
        flip=bool(flip),
        iou=float(iou),
        rgb_size=list(rgb_size),
        th_shape=list(th_shape),
        n_samples=len(samples),
    )


def interp_landmarks(lm, conf, t, fps_rgb):
    n = len(lm)
    f = t * fps_rgb
    i0 = int(np.floor(f))
    w = f - i0
    i1 = i0 + 1
    nearest = int(round(f))
    skew_ms = (nearest / fps_rgb - t) * 1000.0
    out = dict(lm=None, conf=0.0, nearest=int(np.clip(nearest, 0, n - 1)), skew_ms=skew_ms, motion_px=0.0, how="none")
    if i0 < 0 or i1 >= n:
        if 0 <= nearest < n and np.isfinite(lm[nearest]).all():
            out.update(lm=lm[nearest], conf=float(conf[nearest]), how="nearest")
        return out
    a, b = lm[i0], lm[i1]
    oka, okb = np.isfinite(a).all(), np.isfinite(b).all()
    if oka and okb:
        out.update(
            lm=(1 - w) * a + w * b,
            conf=float(min(conf[i0], conf[i1])),
            how="interp",
            motion_px=float(np.max(np.linalg.norm((b - a)[:, :2], axis=1))),
        )
    elif oka or okb:
        k = i0 if oka else i1
        out.update(lm=lm[k], conf=float(conf[k]), how="nearest")
    return out


def estimate_offset(rgb_lm, th_masks, t_th, cfg, max_lag=1.0):
    fps_rgb = cfg["RGB_FPS"]
    ar = np.array(
        [
            cv2.contourArea(cv2.convexHull(l[:, :2].astype(np.float32).reshape(-1, 1, 2)))
            if np.isfinite(l).all()
            else np.nan
            for l in rgb_lm
        ]
    )
    at = np.array([m.mean() for m in th_masks], np.float32)
    if np.isfinite(ar).sum() < 20 or np.nanstd(ar) < 1e-6 or at.std() < 1e-6:
        return 0.0, 0.0
    tr = np.arange(len(ar)) / fps_rgb
    ok = np.isfinite(ar)
    arf = np.interp(tr, tr[ok], ar[ok])
    tt = np.asarray(t_th, float)

    def corr_at(off):
        t = tt + off
        v = (t >= 0) & (t <= tr[-1])
        if v.sum() < 15:
            return -1.0
        x = np.interp(t[v], tr, arf)
        y = at[v]
        if x.std() < 1e-9 or y.std() < 1e-9:
            return -1.0
        return float(np.corrcoef(x, y)[0, 1])

    r0 = corr_at(0.0)
    best = (0.0, r0)
    for off in np.arange(-max_lag, max_lag + 1e-9, 0.02):
        r = corr_at(off)
        if r > best[1]:
            best = (float(off), r)
    return best if (best[1] > 0.7 and best[1] - r0 > 0.05 and abs(best[0]) >= 0.5 / fps_rgb) else (0.0, r0)


# -----------------------------------------------------------------------------
# Section 6: ROI Engine, Quality Flags & Rolling Temporal Features
# -----------------------------------------------------------------------------
def roi_stats(T, mask, cx, cy, r):
    H, W = T.shape
    nan = dict(
        n_px=0, on_skin=0.0, T_mean=np.nan, T_median=np.nan,
        T_min=np.nan, T_max=np.nan, T_p10=np.nan, T_p90=np.nan,
        T_std=np.nan, dT_roi=np.nan,
    )
    if not (np.isfinite(cx) and np.isfinite(cy)):
        return nan
    x0, x1 = int(max(0, np.floor(cx - r))), int(min(W, np.ceil(cx + r) + 1))
    y0, y1 = int(max(0, np.floor(cy - r))), int(min(H, np.ceil(cy + r) + 1))
    if x1 <= x0 or y1 <= y0:
        return nan
    yy, xx = np.mgrid[y0:y1, x0:x1]
    circ = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    skin = circ & mask[y0:y1, x0:x1]
    vals = T[y0:y1, x0:x1][skin].astype(np.float32)
    st = dict(nan)
    st["n_px"] = int(skin.sum())
    st["on_skin"] = float(skin.sum() / max(math.pi * r * r, 1.0))
    if vals.size >= 3:
        st.update(
            T_mean=float(vals.mean()),
            T_median=float(np.median(vals)),
            T_min=float(vals.min()),
            T_max=float(vals.max()),
            T_p10=float(np.percentile(vals, 10)),
            T_p90=float(np.percentile(vals, 90)),
            T_std=float(vals.std()),
            dT_roi=float(vals.max() - vals.min()),
        )
    return st


def rolling_feats(t, y, W):
    n = len(y)
    sd = np.full(n, np.nan)
    pp = np.full(n, np.nan)
    sl = np.full(n, np.nan)
    for i in range(n):
        lo = int(np.searchsorted(t, t[i] - W, side="right"))
        yy = y[lo : i + 1]
        tt = t[lo : i + 1]
        m = np.isfinite(yy)
        if m.sum() >= 3:
            yv, tv = yy[m], tt[m] - tt[m].mean()
            sd[i] = yv.std()
            pp[i] = yv.max() - yv.min()
            sl[i] = float((tv * (yv - yv.mean())).sum() / max((tv ** 2).sum(), 1e-9))
    return sd, pp, sl


def add_temporal(df, cfg):
    df = df.sort_values(["joint_id", "frame"]).reset_index(drop=True)
    for c in ("win_std", "win_ptp", "win_slope_Cps"):
        df[c] = np.nan
    for _, g in df.groupby("joint_id"):
        y = np.where(g["usable"].values, g["T_median"].values, np.nan)
        sd, pp, sl = rolling_feats(g["t_thermal_s"].values, y, cfg["WINDOW_S"])
        df.loc[g.index, "win_std"] = sd
        df.loc[g.index, "win_ptp"] = pp
        df.loc[g.index, "win_slope_Cps"] = sl
    return df


# -----------------------------------------------------------------------------
# Section 10: Overlay Rendering (RGB + MediaPipe & Thermal + Aligned ROIs)
# -----------------------------------------------------------------------------
def read_rgb_frames(path, wanted):
    wanted = set(int(w) for w in wanted)
    if not wanted:
        return {}
    max_w = max(wanted)
    out = {}
    for i, fr in iter_frames(path):
        if i in wanted:
            out[i] = fr
        if i >= max_w:
            break
    return out


def draw_hand(img, pts, rad, color, cfg, roi=True, bone=(255, 255, 255), scale=1.0):
    if not (np.isfinite(pts).all() and np.isfinite(rad)):
        return img
    p = (pts * scale).astype(int)
    for a, b in BONES:
        cv2.line(img, tuple(p[a]), tuple(p[b]), bone, 1, cv2.LINE_AA)
    for j in range(21):
        r = max(2, int(round((cfg["ROI_ALPHA_TIP"] if j in TIPS else cfg["ROI_ALPHA"]) * rad * scale)))
        if roi:
            cv2.circle(img, tuple(p[j]), r, color, 1, cv2.LINE_AA)
        cv2.circle(img, tuple(p[j]), 2, (0, 255, 0), -1, cv2.LINE_AA)
    return img


def thermal_to_colormap(T_frame, lo, hi, H=360):
    g = np.clip((T_frame.astype(np.float32) - lo) / max(hi - lo, 1e-3) * 255, 0, 255).astype(np.uint8)
    img = cv2.applyColorMap(g, cv2.COLORMAP_INFERNO)
    s = H / img.shape[0]
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC), s


def build_overlays(rgb_path, T, t_arr, rgb, pts_arr, rad_arr, df, scale_factor, cfg):
    """Return a list of annotated RGB & Thermal overlay dicts (RGB uint8) for Streamlit display."""
    N = len(T)
    if N == 0:
        return []
    n_show = min(cfg["OVERLAY_FRAMES"], N)
    ks = sorted(set(np.linspace(0, N - 1, n_show).astype(int).tolist()))

    rgb_idx_map = {}
    for k in ks:
        sub = df[df["frame"] == k]
        rgb_idx_map[k] = int(sub["rgb_idx"].iloc[0]) if len(sub) else 0

    rgbf = read_rgb_frames(rgb_path, set(rgb_idx_map.values()))
    lo, hi = np.percentile(T[N // 2].astype(np.float32), [1, 99.5])
    H = 360
    overlays = []

    for k in ks:
        th_bgr, s = thermal_to_colormap(T[k], lo, hi, H=H)
        draw_hand(th_bgr, pts_arr[k], rad_arr[k], (0, 200, 255), cfg, scale=s)

        r_idx = rgb_idx_map[k]
        rg_bgr = rgbf.get(r_idx)
        if rg_bgr is None:
            rg_bgr = np.zeros((H, int(H * 4 / 3), 3), np.uint8)
        else:
            rg_bgr = rg_bgr.copy()
        if rg_bgr.ndim == 2:
            rg_bgr = cv2.cvtColor(rg_bgr, cv2.COLOR_GRAY2BGR)
        sr = H / rg_bgr.shape[0]
        rg_bgr = cv2.resize(rg_bgr, None, fx=sr, fy=sr)
        rsz = np.array(rgb["size"], np.float32)
        if 0 <= r_idx < len(rgb["lm"]):
            lmr = rgb["lm"][r_idx, :, :2] * rsz * sr
            draw_hand(rg_bgr, lmr, rad_arr[k] / max(scale_factor, 1e-6) * sr, (53, 107, 255), cfg)

        row = df[df["frame"] == k]
        usable_cnt = int(row["usable"].sum()) if len(row) else 0
        skew_val = float(row["skew_ms"].iloc[0]) if len(row) and np.isfinite(row["skew_ms"].iloc[0]) else 0.0

        overlays.append(
            dict(
                frame=int(k),
                time_s=round(float(t_arr[k]), 2),
                rgb_img=cv2.cvtColor(rg_bgr, cv2.COLOR_BGR2RGB),
                thermal_img=cv2.cvtColor(th_bgr, cv2.COLOR_BGR2RGB),
                caption=f"Frame #{k} | t = {float(t_arr[k]):.2f}s | RGB #{r_idx} | skew {skew_val:+.1f} ms | Usable joints: {usable_cnt}/21",
            )
        )
    return overlays


# -----------------------------------------------------------------------------
# Main Entry Point for Streamlit (app.py)
# -----------------------------------------------------------------------------
def process_hand_recordings(
    participant_id,
    protocol,
    rgb_path,
    thermal_path=None,
    rad_paths=None,
    sample_every_sec=0.0,
    cal_data=None,
    adv_config=None,
    status_cb=None,
):
    """Run the full RGB-guided radiometric hand-joint thermal pipeline on one recording."""
    def log(msg):
        if status_cb is not None:
            status_cb(msg)

    cfg = dict(DEFAULT_CFG)
    adv_config = adv_config or {}
    if "mp_confidence" in adv_config:
        cfg["MP_DET_CONF"] = float(adv_config["mp_confidence"])
        cfg["MP_TRACK_CONF"] = float(adv_config["mp_confidence"])
    if "roi_radius" in adv_config:
        cfg["MIN_ROI_PX"] = float(adv_config["roi_radius"])

    temp_unit = adv_config.get("temp_unit", "°C")
    unit_offset = 273.15 if temp_unit == "K" else 0.0

    # Auto-detect RGB FPS from file metadata if valid
    rgb_meta = video_meta(rgb_path)
    if rgb_meta["fps"] and 5.0 <= rgb_meta["fps"] <= 120.0:
        cfg["RGB_FPS"] = rgb_meta["fps"]

    log("1/5 Extracting MediaPipe 21-point hand landmarks from RGB video...")
    rgb = extract_rgb_landmarks(rgb_path, cfg)

    log("2/5 Decoding radiometric thermal sequence & aligning timestamps...")
    T_raw, thermal_info = load_thermal(rad_paths, thermal_path, cfg)
    T_all, t_all, fidx_all, align_info = align_thermal(T_raw, len(rgb["lm"]), cfg)

    # Apply sampling interval if requested
    if sample_every_sec and sample_every_sec > 0:
        keep = [0]
        last_t = float(t_all[0])
        for i in range(1, len(t_all)):
            if float(t_all[i]) - last_t >= sample_every_sec - 1e-6:
                keep.append(i)
                last_t = float(t_all[i])
        keep = np.array(keep, dtype=int)
    else:
        keep = np.arange(len(T_all), dtype=int)

    log("3/5 Running RGB-to-thermal camera rig calibration...")
    if cal_data is None:
        cal = calibrate_single(rgb, T_all, t_all, cfg)
    else:
        cal = cal_data

    A_map = make_A(cal["params"], tuple(cal["rgb_size"]), cal["flip"])
    scale_factor = float(np.sqrt(abs(np.linalg.det(A_map[:, :2]))))

    # Subsample thermal arrays for ROI extraction
    T = T_all[keep]
    t_arr = t_all[keep]
    fidx = fidx_all[keep]

    mc = [hand_mask(T[k].astype(np.float32)) for k in range(len(T))]
    masks = [m for m, _ in mc]
    contrasts = np.array([c for _, c in mc], dtype=np.float32)

    offset_s = 0.0
    if protocol == "open_close" and len(T) >= 20:
        offset_s, _ = estimate_offset(rgb["lm"], masks, t_arr, cfg)

    log("4/5 Extracting per-joint thermal ROIs, quality flags & ΔT fluctuations...")
    N, Ht, Wt = T.shape
    rgb_size = tuple(int(v) for v in rgb["size"])
    rows = []
    pts_arr = np.full((N, 21, 2), np.nan, np.float32)
    rad_arr = np.full(N, np.nan, np.float32)

    for k in range(N):
        t_th = float(t_arr[k])
        Tk = T[k].astype(np.float32)
        mask = masks[k]
        fl_frame = set()
        conf_j = np.zeros(21, dtype=np.float32)

        ii = interp_landmarks(rgb["lm"], rgb["conf"], t_th + offset_s, cfg["RGB_FPS"])
        skew, nearest = ii["skew_ms"], ii["nearest"]
        motion_rel = 0.0
        if ii["lm"] is None:
            pts = None
        else:
            pts = warp_pts(ii["lm"][:, :2], rgb_size, A_map)
            conf_j[:] = ii["conf"]
            motion_rel = ii["motion_px"] * rgb_size[0] * scale_factor / max(np.linalg.norm(pts[0] - pts[9]), 1e-6)

        if pts is None or not np.isfinite(pts).all():
            fl_frame.add("NO_HAND")
            pts = np.full((21, 2), np.nan)
        else:
            L = float(np.linalg.norm(pts[0] - pts[9]))
            pts_arr[k] = pts
            if L < 3:
                fl_frame.add("NO_HAND")

        L = float(np.linalg.norm(pts[0] - pts[9])) if "NO_HAND" not in fl_frame else np.nan
        rad_arr[k] = L

        if contrasts[k] < cfg["LOW_CONTRAST_C"]:
            fl_frame.add("LOW_CONTRAST")
        if "NO_HAND" not in fl_frame:
            inside = np.mean(
                [
                    mask[int(np.clip(round(y), 0, Ht - 1)), int(np.clip(round(x), 0, Wt - 1))]
                    if (0 <= x < Wt and 0 <= y < Ht)
                    else False
                    for x, y in pts
                ]
            )
            if inside < 1 - cfg["CALIB_OUTSIDE_FRAC"]:
                fl_frame.add("CALIB_ERROR")
            if abs(skew) > cfg["SKEW_TOL_MS"] or motion_rel > cfg["FAST_MOTION_REL"]:
                fl_frame.add("SYNC_SKEW")

        if "NO_HAND" not in fl_frame:
            pc = pts[[0, 5, 9, 13, 17]].mean(0)
            palm = roi_stats(Tk, mask, pc[0], pc[1], 0.25 * L)
            T_palm = palm["T_median"]
        else:
            T_palm = np.nan

        ext = {}
        tipd = {}
        if "NO_HAND" not in fl_frame:
            for f_name, idx_list in FINGERS.items():
                ext[f_name] = float(
                    np.linalg.norm(pts[idx_list[3]] - pts[0])
                    / (np.linalg.norm(pts[idx_list[0]] - pts[0]) + 1e-6)
                )
            for a in TIPS:
                tipd[a] = min(np.linalg.norm(pts[a] - pts[b]) for b in TIPS if b != a) / max(L, 1e-6)

        for j in range(21):
            fl = set(fl_frame)
            x, y = pts[j] if "NO_HAND" not in fl_frame else (np.nan, np.nan)
            alpha = cfg["ROI_ALPHA_TIP"] if j in TIPS else cfg["ROI_ALPHA"]
            r = max(cfg["MIN_ROI_PX"], alpha * L) if np.isfinite(L) else np.nan
            if "NO_HAND" in fl_frame:
                st_roi = roi_stats(Tk, mask, np.nan, np.nan, 1)
            else:
                st_roi = roi_stats(Tk, mask, x, y, r)
                if not (0 <= x < Wt and 0 <= y < Ht):
                    fl.add("OUT_OF_FRAME")
                if st_roi["on_skin"] < cfg["MIN_ON_SKIN"]:
                    fl.add("ROI_OFF_SKIN")
                if conf_j[j] < cfg["LOW_CONF"]:
                    fl.add("LOW_CONF")
                if j in tipd and tipd[j] < cfg["TIP_OVERLAP_REL"]:
                    fl.add("OCCLUSION")
                if np.isfinite(st_roi["T_median"]) and not (
                    cfg["TEMP_PLAUSIBLE"][0] <= st_roi["T_median"] <= cfg["TEMP_PLAUSIBLE"][1]
                ):
                    fl.add("TEMP_OUT_OF_RANGE")

            fng = next((f_name for f_name, idx_list in FINGERS.items() if j in idx_list), "WRIST")
            row = dict(
                participant_id=participant_id,
                protocol=protocol,
                frame=k,
                npy_idx=int(fidx[k]),
                t_thermal_s=round(t_th, 4),
                rgb_idx=nearest,
                skew_ms=skew,
                joint_id=j,
                joint=JOINTS[j],
                finger=fng,
                x=x,
                y=y,
                roi_r=r,
                hand_len=L,
                conf=float(conf_j[j]),
                finger_ext=ext.get(fng, np.nan),
                T_palm=T_palm,
                dT_vs_palm=(
                    st_roi["T_median"] - T_palm
                    if np.isfinite(T_palm) and np.isfinite(st_roi["T_median"])
                    else np.nan
                ),
                flags="|".join(sorted(fl)),
                usable=not (fl & CRITICAL),
            )
            row.update(st_roi)
            rows.append(row)

    full_df = add_temporal(pd.DataFrame(rows), cfg)

    # Build dashboard summary table & time-series table
    usable_df = full_df[full_df["usable"] & full_df["T_median"].notna()]
    src_df = usable_df if len(usable_df) else full_df[full_df["T_median"].notna()]

    summary_rows = []
    for j_name in JOINTS:
        g_all = full_df[full_df["joint"] == j_name]
        g = src_df[src_df["joint"] == j_name]
        vals = g["T_median"].dropna().values + unit_offset
        usable_pct = round(float(100.0 * g_all["usable"].mean()), 1) if len(g_all) else 0.0
        if len(vals):
            summary_rows.append(
                {
                    "Joint": j_name,
                    f"Mean ({temp_unit})": round(float(np.mean(vals)), 2),
                    f"Min ({temp_unit})": round(float(np.min(vals)), 2),
                    f"Max ({temp_unit})": round(float(np.max(vals)), 2),
                    f"Δ Temp ({temp_unit})": round(float(np.max(vals) - np.min(vals)), 2),
                    f"Std Dev ({temp_unit})": round(float(np.std(vals)), 2),
                    "Usable (%)": usable_pct,
                }
            )
        else:
            summary_rows.append(
                {
                    "Joint": j_name,
                    f"Mean ({temp_unit})": np.nan,
                    f"Min ({temp_unit})": np.nan,
                    f"Max ({temp_unit})": np.nan,
                    f"Δ Temp ({temp_unit})": np.nan,
                    f"Std Dev ({temp_unit})": np.nan,
                    "Usable (%)": usable_pct,
                }
            )
    summary_df = pd.DataFrame(summary_rows)

    piv = (
        src_df.pivot_table(index="t_thermal_s", columns="joint", values="T_median", aggfunc="median")
        .reindex(columns=JOINTS)
        + unit_offset
    )
    piv = piv.round(2).reset_index().rename(columns={"t_thermal_s": "Time (s)"})

    log("5/5 Rendering annotated RGB & Thermal overlay images...")
    overlays = build_overlays(rgb_path, T, t_arr, rgb, pts_arr, rad_arr, full_df, scale_factor, cfg)

    return dict(
        participant_id=participant_id,
        protocol=protocol,
        unit=temp_unit,
        frames_processed=int(N),
        summary_df=summary_df,
        time_series=piv,
        full_features_df=full_df,
        rig_calibration=cal,
        overlays=overlays,
        meta=dict(
            thermal_unit=thermal_info["unit"],
            thermal_absolute=thermal_info["absolute"],
            offset_s=round(float(offset_s), 3),
            calib_iou=round(float(cal.get("iou", 0.0)), 3),
            usable_pct=round(float(100.0 * full_df["usable"].mean()), 1) if len(full_df) else 0.0,
            timebase_note=align_info["note"],
        ),
    )
