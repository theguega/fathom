"""The live viewer, in about forty lines.

You own the loop here exactly as you do in Rust: `poll()` pumps window events
and returns False once the window is closed, and nothing calls back into your
code.

Run with `python scrub.py` after `maturin develop` in crates/fathom-py.
"""

import time

import numpy as np

import fathom

W, H = 160, 120

r = fathom.Renderer.window("fathom - scrub", 1280, 720)
r.set_clear_color((18, 18, 22, 255))

tex = r.upload_texture(bytes(W * H * 4), W, H)
orbit = fathom.Orbit(1.6)

ee_path = np.zeros((0, 3), np.float32)
start = time.monotonic()

while r.poll():
    t = time.monotonic() - start

    # The producer. In a real integration this is your MCAP or LeRobot adapter;
    # fathom owns no threads and no buffering, so a slow one just means a stale
    # frame rather than a stalled loop.
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    wave = (np.sin(xs * 0.05 + t * 2.0) * np.cos(ys * 0.05 - t) + 1.0) * 0.5
    frame = np.stack(
        [wave * 255, xs / W * 255, ys / H * 255, np.full_like(wave, 255)], axis=-1
    ).astype(np.uint8)
    tex.update(frame.tobytes())

    tip = np.array([[np.cos(t * 0.7) * 0.4, np.sin(t * 1.3) * 0.15 + 0.2, np.sin(t * 0.7) * 0.4]], np.float32)
    ee_path = np.ascontiguousarray(np.vstack([ee_path, tip])[-512:])

    r.begin_frame()

    # Layout is Rect maths in your code; there is no panel manager.
    x, y, w, h = r.viewport
    left = (x, y, w / 2, h)
    right = (x + w / 2, y, w / 2, h)

    side = min(left[2] - 32, (left[3] - 32) * tex.aspect)
    fathom.draw_texture(r, tex, (left[0] + 16, left[1] + 16, side, side / tex.aspect))
    fathom.draw_text_at(r, (left[0] + 16, left[1] + 16), f"t = {t:6.2f}s", 2)

    orbit.rotate(0.004, 0.0)
    r.scene(orbit.camera(right[2] / right[3]), right)
    fathom.draw_grid(r, 20, 0.1, (45, 45, 55, 255))
    if len(ee_path) > 1:
        # Per-vertex colour is the whole "age gradient" feature, in caller code.
        age = np.linspace(0, 1, len(ee_path), dtype=np.float32)
        fathom.draw_line_strip_3d_vc(r, ee_path, fathom.colormap(age, 0, 1, "viridis"))
    r.end_scene()

    r.end_frame()
