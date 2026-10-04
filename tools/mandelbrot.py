#!/usr/bin/env python3
"""Render the Mandelbrot set to a PNG.

Escape-time algorithm, vectorized with NumPy, with smooth (continuous)
coloring so the boundary shows clean bands instead of contour steps.
Requires numpy and Pillow.

Examples:
    python3 tools/mandelbrot.py
    python3 tools/mandelbrot.py --width 1920 --height 1200 --iter 1024 --bands 12
    python3 tools/mandelbrot.py --cx -0.744 --cy 0.114 --scale 0.05 --iter 4000 --palette ember
    python3 tools/mandelbrot.py --cx -0.0012 --cy 0 --scale 0.004 --iter 8000 --palette ice
"""
from __future__ import annotations

import argparse
import pathlib
import time

import numpy as np
from PIL import Image

# Cosine palettes: color(t) = a + b * cos(2*pi*(c*t + d)), t in [0, 1].
# `c` is multiplied by --bands at render time, so each entry keeps c = 1.
PALETTES: dict[str, dict] = {
    "classic": dict(a=[0.50, 0.50, 0.50], b=[0.50, 0.50, 0.50], c=[1.0, 1.0, 1.0], d=[0.00, 0.10, 0.20]),
    "rainbow": dict(a=[0.50, 0.50, 0.50], b=[0.50, 0.50, 0.50], c=[1.0, 1.0, 1.0], d=[0.00, 0.33, 0.67]),
    "ember":   dict(a=[0.55, 0.40, 0.30], b=[0.45, 0.45, 0.35], c=[1.0, 1.0, 1.0], d=[0.00, 0.12, 0.28]),
    "ice":     dict(a=[0.35, 0.45, 0.55], b=[0.40, 0.40, 0.45], c=[1.0, 1.0, 1.0], d=[0.55, 0.60, 0.70]),
}


def make_grid(width: int, height: int, cx: float, cy: float, scale: float) -> np.ndarray:
    """Complex-coordinate grid with square pixels.

    `scale` is the real-axis span of the view; the imaginary span is scaled
    by height/width so pixels stay square.
    """
    xs = np.linspace(cx - scale / 2, cx + scale / 2, width)
    yspan = scale * height / width
    ys = np.linspace(cy - yspan / 2, cy + yspan / 2, height)
    return (xs[None, :] + 1j * ys[:, None]).astype(np.complex128)


def escape_time(c: np.ndarray, max_iter: int) -> np.ndarray:
    """Smooth iteration count for every point of the grid.

    Returns float64: 0.0 for points that never escape (the set itself),
    otherwise nu = (n + 1) - log2(log|z_n|), the standard continuous
    extension of the escape-time count, where n is the escape iteration.
    """
    # Work on the raveled grid so the function accepts any input shape.
    cf = c.ravel()
    z = np.zeros(cf.size, np.complex128)
    smooth = np.zeros(cf.size, np.float64)
    mask = np.ones(cf.size, np.bool_)
    for i in range(max_iter):
        if not mask.any():
            break
        idx = np.flatnonzero(mask)
        z[idx] = z[idx] * z[idx] + cf[idx]
        mag2 = z[idx].real**2 + z[idx].imag**2
        escaped = mag2 > 4.0
        if escaped.any():
            ei = idx[escaped]
            m2 = mag2[escaped]
            smooth[ei] = (i + 2.0) - np.log2(0.5 * np.log(m2))
            keep = ~escaped
            mask = np.zeros(cf.size, np.bool_)
            mask[idx[keep]] = True
    return smooth.reshape(c.shape)


def render(nu: np.ndarray, palette: dict, bands: int, max_iter: int) -> np.ndarray:
    """Map smooth counts to an HxWx3 uint8 RGB array."""
    a = np.asarray(palette["a"], np.float64)
    b = np.asarray(palette["b"], np.float64)
    cp = np.asarray(palette["c"], np.float64) * max(1, bands)
    d = np.asarray(palette["d"], np.float64)
    t = np.clip(nu / max(1, max_iter), 0.0, 1.0)
    theta = 2.0 * np.pi * (cp * t[..., None] + d[None, None, :])
    rgb = 255.0 * np.clip(a + b * np.cos(theta), 0.0, 1.0)
    rgb = rgb.astype(np.uint8)
    rgb[nu == 0] = 0  # interior of the set: black
    return rgb


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Render the Mandelbrot set to a PNG.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples:")[1],
    )
    ap.add_argument("--width", type=int, default=1200)
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--cx", type=float, default=-0.6, help="view center, real part")
    ap.add_argument("--cy", type=float, default=0.0, help="view center, imaginary part")
    ap.add_argument("--scale", type=float, default=3.0, help="real-axis span of the view")
    ap.add_argument("--iter", dest="max_iter", type=int, default=512, help="max iterations")
    ap.add_argument("--bands", type=int, default=8, help="color cycles across the iteration range")
    ap.add_argument("--palette", choices=sorted(PALETTES), default="classic")
    ap.add_argument("--output", default="artifacts/mandelbrot.png")
    args = ap.parse_args()

    c = make_grid(args.width, args.height, args.cx, args.cy, args.scale)

    t0 = time.perf_counter()
    nu = escape_time(c, args.max_iter)
    t1 = time.perf_counter()
    rgb = render(nu, PALETTES[args.palette], args.bands, args.max_iter)
    t2 = time.perf_counter()

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(out)

    in_set = (nu == 0).mean() * 100
    print(f"saved {out} ({args.width}x{args.height})")
    print(f"  palette={args.palette} bands={args.bands} max_iter={args.max_iter}")
    print(f"  compute {t1 - t0:.2f}s  color {t2 - t1:.2f}s  total {t2 - t0:.2f}s")
    print(f"  in-set pixels: {in_set:.1f}%   max smooth count: {nu.max():.1f}")


if __name__ == "__main__":
    main()
