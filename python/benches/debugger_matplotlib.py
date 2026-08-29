"""The same multi-panel debugger as debugger_fathom.py, in matplotlib.

Kept beside its counterpart so the README's "identical code length, 65x the
frame rate" claim can be checked rather than taken on faith. Both files are 32
significant lines: a camera stream, a 3D end-effector path, and a scrolling
joint plot, all updated every frame.

This one is written the fast way - `set_data` on existing artists, not a fresh
figure per frame - so the comparison is against matplotlib at its best.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

CAM_W, CAM_H, N = 160, 120, 512
fig = plt.figure(figsize=(12.8, 7.2))
ax_img = fig.add_subplot(221)
ax_3d = fig.add_subplot(122, projection="3d")
ax_ts = fig.add_subplot(223)

im = ax_img.imshow(np.zeros((CAM_H, CAM_W, 3), np.uint8))
ax_img.set_axis_off()
path = np.zeros((N, 3), np.float32)
line3d, = ax_3d.plot(path[:, 0], path[:, 1], path[:, 2], lw=0.8)
ax_3d.set_xlim(-0.5, 0.5); ax_3d.set_ylim(-0.5, 0.5); ax_3d.set_zlim(-0.5, 0.5)
series = np.zeros(N, np.float32)
line_ts, = ax_ts.plot(series, color="green")
ax_ts.set_ylim(-1, 1)
title = ax_img.set_title("t = 0.00s")

def update(i):
    t = i * 0.016
    ys, xs = np.mgrid[0:CAM_H, 0:CAM_W].astype(np.float32)
    frame = np.stack([np.sin(xs * .05 + t) * 127 + 128,
                      xs / CAM_W * 255, ys / CAM_H * 255], -1).astype(np.uint8)
    im.set_data(frame)
    path[:-1] = path[1:]
    path[-1] = [np.cos(t * .7) * .4, np.sin(t * 1.3) * .15, np.sin(t * .7) * .4]
    line3d.set_data_3d(path[:, 0], path[:, 1], path[:, 2])
    series[:-1] = series[1:]; series[-1] = np.sin(t)
    line_ts.set_ydata(series)
    title.set_text(f"t = {t:.2f}s")
    return im, line3d, line_ts, title

anim = FuncAnimation(fig, update, interval=16, blit=False, cache_frame_data=False)
plt.show()
