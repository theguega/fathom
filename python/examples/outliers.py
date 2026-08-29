"""The Rust outliers example, using scipy for the distance metric.

Same shape, same call names, which is the point: the binding is a mirror, not a
reinterpretation. fathom knows nothing about outliers - you compute a score per
demo and map it through `colormap`.

Run with `python outliers.py`.
"""

import numpy as np

import fathom

rng = np.random.default_rng(0)

# Twenty demonstrations of the same reach, a few of them sloppy.
steps = np.linspace(0, 1, 120, dtype=np.float32)
demos = []
for d in range(20):
    wobble = 0.25 if d % 7 == 0 else 0.03
    phase = d * 0.7
    demos.append(
        np.stack(
            [
                steps * 1.2 - 0.6,
                np.sin(steps * 3 + phase) * wobble + 0.3,
                np.cos(steps * 5 + phase) * wobble,
            ],
            axis=-1,
        ).astype(np.float32)
    )

reference = np.stack(
    [steps * 1.2 - 0.6, np.full_like(steps, 0.3), np.zeros_like(steps)], axis=-1
).astype(np.float32)

# Your metric, your code. Swap in scipy.spatial.distance or a DTW and nothing
# below changes.
scores = np.array([np.linalg.norm(d - reference, axis=1).mean() for d in demos], np.float32)
colors = fathom.colormap(scores, 0.0, float(scores.max()), "turbo")

r = fathom.Renderer.window("fathom - outliers", 1100, 800)
r.set_clear_color((18, 18, 22, 255))
orbit = fathom.Orbit(2.0)

while r.poll():
    r.begin_frame()
    r.scene(orbit.camera(r.aspect))
    fathom.draw_grid(r, 20, 0.1, (40, 40, 50, 255))
    for demo, c in zip(demos, colors):
        fathom.draw_line_strip_3d(r, demo, (int(c[0]), int(c[1]), int(c[2]), 140))
    fathom.draw_line_strip_3d(r, reference, (255, 255, 255, 255))
    r.end_scene()
    fathom.draw_text_at(r, (12, 12), "20 demos, coloured by distance (Turbo)", 1, (150, 150, 160, 255))
    r.end_frame()
