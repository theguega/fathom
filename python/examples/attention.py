"""A cross-attention map, alpha-blended over the camera frame.

The whole path is: your model's attention weights -> `colormap` -> a texture ->
`draw_texture` with an alpha tint. fathom has no notion of attention, no
heatmap primitive and no overlay system; a heatmap is a texture, and blending
one over another is the tint argument you already have.

Run with `python attention.py`.
"""

import numpy as np

import fathom

CAM_W, CAM_H = 320, 240
ATTN = 16  # the model's patch grid


def camera_frame(t: float) -> np.ndarray:
    """Stand-in for the wrist camera. Yours comes from an adapter."""
    ys, xs = np.mgrid[0:CAM_H, 0:CAM_W].astype(np.float32)
    r = (np.sin(xs * 0.03 + t) * 0.5 + 0.5) * 180
    g = (np.cos(ys * 0.03 - t * 0.7) * 0.5 + 0.5) * 140
    b = np.full_like(r, 90.0)
    return np.stack([r, g, b, np.full_like(r, 255)], -1).astype(np.uint8)


def attention_map(t: float) -> np.ndarray:
    """Stand-in for one head's cross-attention over image patches."""
    ys, xs = np.mgrid[0:ATTN, 0:ATTN].astype(np.float32)
    cx, cy = ATTN / 2 + np.cos(t) * 4, ATTN / 2 + np.sin(t * 1.3) * 4
    d = np.hypot(xs - cx, ys - cy)
    a = np.exp(-(d**2) / 8.0)
    return (a / a.max()).astype(np.float32)


r = fathom.Renderer.window("fathom - attention", 1000, 760)
r.set_clear_color((18, 18, 22, 255))

camera = r.upload_texture(camera_frame(0.0).tobytes(), CAM_W, CAM_H)
# Nearest, so the patch grid stays legible instead of being smoothed into mush.
heat = r.upload_texture(bytes(ATTN * ATTN * 4), ATTN, ATTN, nearest=True)

t = 0.0
while r.poll():
    t += 0.016
    camera.update(camera_frame(t).tobytes())

    # Attention weights are just scalars: the same `colormap` that serves depth
    # maps and similarity matrices serves this.
    weights = attention_map(t)
    heat.update(fathom.colormap(weights.ravel(), 0.0, 1.0, "magma").tobytes())

    r.begin_frame()
    x, y, w, h = r.viewport
    panel = (x + 16, y + 16, w - 32, h - 32)
    side = min(panel[2], panel[3] * (CAM_W / CAM_H))
    dst = (panel[0], panel[1], side, side * CAM_H / CAM_W)

    fathom.draw_texture(r, camera, dst)
    # The overlay is the same call with an alpha tint. No blend modes, no
    # compositor, no overlay trait: alpha is a channel on the colour you pass.
    fathom.draw_texture(r, heat, dst, (255, 255, 255, 140))
    fathom.draw_bbox(r, dst, (90, 90, 100, 255))

    peak = np.unravel_index(weights.argmax(), weights.shape)
    fathom.draw_text_at(r, (dst[0], dst[1] + dst[3] + 10),
                        f"peak patch ({peak[1]:>2}, {peak[0]:>2})  magma  alpha 0.55", 2)
    r.end_frame()
