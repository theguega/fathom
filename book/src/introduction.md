# fathom

**A low-level, zero-overhead visualization library for multimodal spatiotemporal
data.**

Video, images, depth, point clouds, trajectories, kinematics, text and time
series, through one set of composable drawing primitives and no framework.

fathom operates on visualization only: pixels, points and transforms in, pixels
out. It is aimed at VLA and robot policy debugging, autonomous driving logs,
world models, and any pipeline where you need to *see* what a model did across
time and space.

## Thirty seconds

```rust,no_run
use fathom::prelude::*;

# fn frame(ctx: &mut Ctx, orbit: &Orbit, camera: &Texture, path: &[WorldPoint]) {
let mut f = begin_frame(ctx);
let [left, right] = f.viewport().split_h();

// 2D: the camera stream, letterboxed rather than stretched.
draw_texture(&mut f, camera, left.fit_aspect(camera.aspect()), Color::WHITE);
draw_text_at(&mut f, ImagePoint::new(8.0, 8.0), "episode 41", TextScale::X2, Color::WHITE);

// 3D: the same instant, in world space, in the other half of the window.
let mut s = f.scene_in(&orbit.camera(right.w / right.h), right);
draw_grid(&mut s, 20, Meters(0.1), Color::GRAY);
draw_line_strip_3d(&mut s, path, Color::GREEN);
s.end();

f.end();
# }
```

## What is not here

No plotting subsystem: a time series is `draw_line_strip_2d` per channel. No
panel manager: layout is `Rect` maths in your code. No timeline widget, no
overlay trait objects, no analysis crate, no kinematics, and no
depth-to-point-cloud conversion.

Each of those absences is explained in [Philosophy](./philosophy.md), and none
of them is an omission to be filled in later.
