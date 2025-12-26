"""drum2d.simulate

Synthetic video generator for a 2D rotating-drum end-face transparent window.

Generates:
- video.mp4 (BGR, HxW)
- gt.npz (theta_gt, surface_xy, regime_id, events, cfg_json)
- meta.json (config + reproducibility)
- preview.png (one frame with GT overlays)

Only uses numpy/opencv/scipy/matplotlib/tqdm + stdlib.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Support both:
# - package run: python -m drum2d.main ...
# - script run inside drum2d/: python main.py ...
try:
    from .utils import (
        circle_mask,
        circle_y_bounds_for_x,
        clamp,
        deg2rad,
        ensure_dir,
        line_from_theta,
        set_global_seed,
        write_json,
    )
except ImportError:  # pragma: no cover
    from utils import (
        circle_mask,
        circle_y_bounds_for_x,
        clamp,
        deg2rad,
        ensure_dir,
        line_from_theta,
        set_global_seed,
        write_json,
    )


@dataclass
class SimConfig:
    """Simulation configuration."""

    seconds: float = 8.0
    fps: int = 30
    rpm: float = 25.0
    fill: float = 0.55

    noise_sigma: float = 8.0
    blur_sigma: float = 1.0
    glare_strength: float = 0.15

    regime: str = "auto"  # auto|slipping|cascading|cataracting
    enable_events: bool = True

    w: int = 512
    h: int = 512
    r: float = 220.0

    # surface sampling
    surface_m: int = 200


def _resolve_regime(regime: str, rpm: float) -> Tuple[str, int]:
    regime = str(regime).lower()
    if regime == "auto":
        if rpm < 15:
            regime = "slipping"
        elif rpm < 35:
            regime = "cascading"
        else:
            regime = "cataracting"

    if regime not in {"slipping", "cascading", "cataracting"}:
        raise ValueError(f"Unknown regime: {regime}")

    regime_id = {"slipping": 0, "cascading": 1, "cataracting": 2}[regime]
    return regime, int(regime_id)


def _theta_base(fill: float, regime: str) -> float:
    # fill -> higher fill tends to slightly larger angle
    fill = clamp(fill, 0.05, 0.95)
    theta = 20.0 + 25.0 * fill  # 21..44 deg
    if regime == "slipping":
        theta -= 3.0
    elif regime == "cataracting":
        theta += 4.0
    return float(clamp(theta, 15.0, 50.0))


def _surface_curve(
    xs: np.ndarray,
    t: float,
    cfg: SimConfig,
    rng: np.random.Generator,
    theta0: float,
    regime: str,
    cx: float,
    cy: float,
    r: float,
    smooth_state: Dict[str, np.ndarray],
    events: List[float],
) -> Tuple[np.ndarray, float]:
    """Compute surface y(x,t) within circle and theta_gt(t) in degrees."""

    # regime-specific oscillation
    if regime == "slipping":
        amp = 1.5
        freq = 0.6
        wav_amp = 3.5
        wav_k = 2.0
        noise_theta = 0.25
    elif regime == "cascading":
        amp = 3.0
        freq = 1.0
        wav_amp = 5.0
        wav_k = 2.5
        noise_theta = 0.35
    else:  # cataracting
        amp = 6.0
        freq = 1.5
        wav_amp = 8.0
        wav_k = 3.0
        noise_theta = 0.6

    osc = amp * math.sin(2.0 * math.pi * freq * t)
    theta_gt = theta0 + osc + rng.normal(0.0, noise_theta)

    k_line = math.tan(deg2rad(theta_gt))

    # y0: choose intercept so that fill roughly matches desired fill
    # This is not physically exact; we map fill to a vertical shift.
    # Lower fill -> surface lower (larger y).
    y0 = cy + (0.65 - cfg.fill) * 160.0
    y_line = y0 + (xs - cx) * k_line

    # smooth random field along x (persistent over time)
    if "smooth_x" not in smooth_state:
        smooth_state["smooth_x"] = np.zeros_like(xs, dtype=np.float32)

    # Update smooth noise slowly to keep continuity
    raw = rng.normal(0.0, 1.0, size=xs.shape).astype(np.float32)
    raw2 = cv2.GaussianBlur(raw.reshape(1, -1), (0, 0), sigmaX=12.0, sigmaY=0.0).reshape(-1)
    smooth_state["smooth_x"] = 0.92 * smooth_state["smooth_x"] + 0.08 * raw2
    smooth = smooth_state["smooth_x"]

    # sinusoidal ripple + smooth noise
    phi = 2.0 * math.pi * 0.25 * t
    delta = wav_amp * np.sin(2.0 * math.pi * wav_k * xs / cfg.w + phi) + 4.0 * smooth

    y = y_line + delta

    # cataracting events: local gaussian bump (upwards => smaller y)
    if regime == "cataracting" and cfg.enable_events:
        # create a few events stochastically
        if rng.random() < 0.015:  # per-frame chance
            events.append(float(t))

        for te in events[-6:]:  # only recent ones impact
            dt = t - te
            if 0.0 <= dt <= 0.35:
                x0 = cx + rng.normal(0.0, 70.0)
                sigma_x = 35.0 + 25.0 * rng.random()
                amp_px = 16.0 * math.exp(-dt / 0.10)
                y += -amp_px * np.exp(-0.5 * ((xs - x0) / sigma_x) ** 2)

    # clip to circle boundaries for each x
    y_top, y_bot = circle_y_bounds_for_x(xs, cx, cy, r)
    y = np.clip(y, y_top + 1.0, y_bot - 1.0)
    return y.astype(np.float32), float(theta_gt)


def _make_glare_mask(h: int, w: int, cx: float, cy: float, r: float) -> np.ndarray:
    """Create a fixed arc-like glare mask in [0,1]."""
    mask = np.zeros((h, w), dtype=np.float32)
    # draw an ellipse arc into mask
    center = (int(round(cx + 35)), int(round(cy - 55)))
    axes = (int(round(r * 0.65)), int(round(r * 0.35)))
    cv2.ellipse(mask, center, axes, 20.0, 205.0, 320.0, 1.0, thickness=10, lineType=cv2.LINE_AA)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=8.0)
    mask = np.clip(mask, 0.0, 1.0)
    return mask


def _init_texture(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Base texture field in [0,1]."""
    n0 = rng.random((h, w), dtype=np.float32)
    n1 = cv2.GaussianBlur(n0, (0, 0), 1.2)
    n2 = cv2.GaussianBlur(n0, (0, 0), 3.5)
    n3 = cv2.GaussianBlur(n0, (0, 0), 9.0)
    tex = 0.55 * n1 + 0.30 * n2 + 0.15 * n3

    # add random dots (grains)
    dots = np.zeros((h, w), dtype=np.float32)
    n_dots = int(2500 + 2000 * rng.random())
    xs = rng.integers(0, w, size=n_dots)
    ys = rng.integers(0, h, size=n_dots)
    rs = rng.integers(1, 4, size=n_dots)
    for x, y, r0 in zip(xs, ys, rs):
        cv2.circle(dots, (int(x), int(y)), int(r0), float(0.9 + 0.2 * rng.random()), thickness=-1, lineType=cv2.LINE_AA)
    dots = cv2.GaussianBlur(dots, (0, 0), 0.6)
    tex = np.clip(tex + 0.65 * dots, 0.0, 1.0)
    return tex.astype(np.float32)


def simulate_case(out_dir: str | Path, cfg: SimConfig, seed: int = 0) -> Path:
    """Simulate one case to out_dir.

    Args:
        out_dir: output folder path.
        cfg: SimConfig.
        seed: random seed for reproducibility.

    Returns:
        Path to out_dir.
    """

    out_dir = ensure_dir(out_dir)
    rng = set_global_seed(seed)

    regime, regime_id = _resolve_regime(cfg.regime, cfg.rpm)

    w, h = int(cfg.w), int(cfg.h)
    cx, cy, r = w / 2.0, h / 2.0, float(cfg.r)
    T = int(round(float(cfg.seconds) * int(cfg.fps)))
    fps = int(cfg.fps)

    # Precompute common masks and components
    c_mask = circle_mask(h, w, cx, cy, r)
    glare_mask = _make_glare_mask(h, w, cx, cy, r)

    # Surface x samples
    xs_surf = np.linspace(cx - r + 2, cx + r - 2, int(cfg.surface_m), dtype=np.float32)

    # texture state (persistent)
    tex = _init_texture(h, w, rng)

    # video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_path = out_dir / "video.mp4"
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("Failed to open VideoWriter. Try installing codecs or use mp4v.")

    theta0 = _theta_base(cfg.fill, regime)
    smooth_state: Dict[str, np.ndarray] = {}
    events: List[float] = []

    theta_gt = np.zeros((T,), dtype=np.float32)
    surface_xy = np.zeros((T, int(cfg.surface_m), 2), dtype=np.float32)
    regime_ids = np.full((T,), int(regime_id), dtype=np.int32)

    preview_frame = None
    preview_pts = None

    # For fast dist computation per x (integer columns)
    x_int = np.arange(w, dtype=np.float32)

    # flow parameters
    v0 = 18.0 + 0.35 * cfg.rpm  # pixels/sec
    d_flow = 12.0 if regime != "cataracting" else 18.0

    for i in range(T):
        t = i / float(fps)

        # surface curve evaluated at sampled xs
        ys_surf, theta_t = _surface_curve(
            xs=xs_surf,
            t=t,
            cfg=cfg,
            rng=rng,
            theta0=theta0,
            regime=regime,
            cx=cx,
            cy=cy,
            r=r,
            smooth_state=smooth_state,
            events=events,
        )
        theta_gt[i] = float(theta_t)
        surface_xy[i, :, 0] = xs_surf
        surface_xy[i, :, 1] = ys_surf

        # approximate surface y for each integer x via interpolation
        y_surface_x = np.interp(x_int, xs_surf, ys_surf).astype(np.float32)

        # compute dist only inside circle
        yy, xx = np.mgrid[0:h, 0:w]
        xx_f = xx.astype(np.float32)
        yy_f = yy.astype(np.float32)
        dist = yy_f - y_surface_x[xx]
        powder = (dist >= 0.0) & c_mask

        # flow direction along surface line
        k = math.tan(deg2rad(theta_t))
        dir_x, dir_y = 1.0, k
        norm = math.hypot(dir_x, dir_y) + 1e-6
        dir_x /= norm
        dir_y /= norm

        # speed field (pixels/frame)
        speed = (v0 / float(fps)) * np.exp(-np.maximum(dist, 0.0) / float(d_flow))
        speed = speed.astype(np.float32)

        dx = (dir_x * speed).astype(np.float32)
        dy = (dir_y * speed).astype(np.float32)

        # remap previous texture to advect powder slightly
        map_x = (xx_f - dx).astype(np.float32)
        map_y = (yy_f - dy).astype(np.float32)
        tex_adv = cv2.remap(tex, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        # inject small fresh noise to avoid overly frozen texture
        inj = rng.normal(0.0, 0.02, size=(h, w)).astype(np.float32)
        tex = np.clip(0.985 * tex_adv + 0.015 * np.clip(tex_adv + inj, 0.0, 1.0), 0.0, 1.0)

        # build intensity image
        air_level = 205.0
        powder_level = 85.0
        img = np.full((h, w), 0.0, dtype=np.float32)
        img[c_mask & (~powder)] = air_level + 25.0 * (tex[c_mask & (~powder)] - 0.5)
        img[powder] = powder_level + 55.0 * (tex[powder] - 0.5)

        # glare
        if cfg.glare_strength > 0:
            img[c_mask] = img[c_mask] + float(cfg.glare_strength) * 255.0 * glare_mask[c_mask]

        # mild brightness drift
        drift = 6.0 * math.sin(2.0 * math.pi * 0.08 * t + 0.7)
        img[c_mask] = img[c_mask] + drift

        # blur + noise
        if cfg.blur_sigma and cfg.blur_sigma > 0:
            img = cv2.GaussianBlur(img, (0, 0), float(cfg.blur_sigma))
        if cfg.noise_sigma and cfg.noise_sigma > 0:
            img = img + rng.normal(0.0, float(cfg.noise_sigma), size=(h, w)).astype(np.float32)

        img = np.clip(img, 0.0, 255.0).astype(np.uint8)

        frame_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # outside circle is black
        frame_bgr[~c_mask] = 0

        writer.write(frame_bgr)

        if preview_frame is None and i == T // 2:
            preview_frame = frame_bgr.copy()
            preview_pts = np.stack([xs_surf, ys_surf], axis=1)

    writer.release()

    # save gt
    gt_path = out_dir / "gt.npz"
    np.savez_compressed(
        gt_path,
        theta_gt=theta_gt.astype(np.float32),
        surface_xy=surface_xy.astype(np.float32),
        regime_id=regime_ids.astype(np.int32),
        events=np.asarray(events, dtype=np.float32),
        cfg_json=json.dumps(asdict(cfg), ensure_ascii=False),
    )

    meta = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": int(seed),
        "cfg": asdict(cfg),
        "resolved": {"regime": regime, "regime_id": int(regime_id)},
        "video": {"path": "video.mp4", "fourcc": "mp4v"},
    }
    write_json(out_dir / "meta.json", meta)

    # preview
    if preview_frame is not None and preview_pts is not None:
        try:
            from .utils import draw_circle_boundary, draw_polyline
        except ImportError:  # pragma: no cover
            from utils import draw_circle_boundary, draw_polyline

        draw_circle_boundary(preview_frame, cx, cy, r, color=(0, 255, 255), thickness=2)
        draw_polyline(preview_frame, preview_pts, color=(0, 0, 255), thickness=2)
        cv2.imwrite(str(out_dir / "preview.png"), preview_frame)

    return out_dir
