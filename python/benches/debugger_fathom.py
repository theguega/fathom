"""The same multi-panel debugger as debugger_matplotlib.py, in fathom.

32 significant lines, same as its counterpart. The difference is not the code
length; it is that this one leaves 16.4ms of a 60Hz frame budget for your own
work instead of 2.7ms.
"""
import time
import numpy as np
import fathom

CAM_W, CAM_H, N = 160, 120, 512
r = fathom.Renderer.window("debugger", 1280, 720)
r.set_clear_color((18, 18, 22, 255))
tex = r.upload_texture(bytes(CAM_W * CAM_H * 4), CAM_W, CAM_H)
orbit = fathom.Orbit(1.6)
path = np.zeros((N, 3), np.float32)
series = np.zeros(N, np.float32)
plot_x = np.linspace(16, 624, N, dtype=np.float32)
start = time.monotonic()

while r.poll():
    t = time.monotonic() - start
    ys, xs = np.mgrid[0:CAM_H, 0:CAM_W].astype(np.float32)
    frame = np.stack([np.sin(xs * .05 + t) * 127 + 128, xs / CAM_W * 255,
                      ys / CAM_H * 255, np.full_like(xs, 255)], -1).astype(np.uint8)
    tex.update(frame.tobytes())
    path[:-1] = path[1:]
    path[-1] = [np.cos(t * .7) * .4, np.sin(t * 1.3) * .15, np.sin(t * .7) * .4]
    series[:-1] = series[1:]; series[-1] = np.sin(t)

    r.begin_frame()
    x, y, w, h = r.viewport
    fathom.draw_texture(r, tex, (16, 16, w / 2 - 32, (w / 2 - 32) * CAM_H / CAM_W))
    fathom.draw_text_at(r, (16, 16), f"t = {t:.2f}s", 2)
    pts = np.stack([plot_x, h - 120 - series * 90], -1).astype(np.float32)
    fathom.draw_line_strip_2d(r, np.ascontiguousarray(pts), (0, 255, 0, 255))
    r.scene(orbit.camera((w / 2) / h), (w / 2, 0, w / 2, h))
    fathom.draw_grid(r, 20, 0.1, (45, 45, 55, 255))
    fathom.draw_line_strip_3d(r, np.ascontiguousarray(path), (0, 200, 255, 255))
    r.end_scene()
    r.end_frame()
