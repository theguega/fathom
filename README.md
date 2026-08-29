# fathom

**A low-level, zero-overhead visualization library for multimodal spatiotemporal data.**

Video, images, depth, point clouds, trajectories, kinematics, text, time series:
one set of composable drawing primitives, no framework.

fathom operates on visualization only. Pixels, points and transforms in, pixels
out. Data ingest (MCAP, LeRobot, ROS bags, video containers) lives in separate
adapter crates that depend on `fathom-core` types alone.

```rust,no_run
use fathom::prelude::*;

# fn frame(ctx: &mut Ctx, orbit: &Orbit, camera: &Texture, path: &[WorldPoint]) {
let mut f = begin_frame(ctx);

// 2D: the camera stream, letterboxed into the left half.
let [left, right] = f.viewport().split_h();
draw_texture(&mut f, camera, left.fit_aspect(camera.aspect()), Color::WHITE);
draw_text_at(&mut f, ImagePoint::new(8.0, 8.0), "episode 41", TextScale::X2, Color::WHITE);

// 3D: the same frame, with a camera bound.
let mut s = f.scene(&orbit.camera(right.w / right.h));
draw_grid(&mut s, 20, Meters(0.1), Color::GRAY);
draw_line_strip_3d(&mut s, path, Color::GREEN);
s.end();

f.end();
# }
```

## The design

Every primitive is a free function over plain slices. The caller owns all state
and owns the loop; the library keeps no registry and never calls back into your
code. Multi-panel layout is `Rect` math in your code, analysis is your stats
crate, and a heatmap is a texture.

Two pipelines carry everything: textured quads and lines. Video is a quad, a
depth heatmap is a quad, text is quads, a point cloud is quads, a trajectory is
lines. The vertex arena is sized once at startup and never reallocates.

Coordinate frames are types. `Point<World>`, `Point<Camera>`, `Point<Image>` and
`Point<Plane>` are `#[repr(transparent)]` over `Vec2`/`Vec3` with a zero-sized
tag, so an extrinsics mix-up is a compile error rather than an overlay that is
subtly wrong in a way you notice three days later.

## The budget, as a number

Measured on an M-series laptop, `cargo bench`:

| | |
|---|---|
| Pack 100k vertices (`draw_line_strip_3d`) | 350 µs |
| Pack 100k vertices (`draw_points_3d`, billboarded) | 389 µs |
| Pack 1k glyphs of text | 53 µs |
| `project` 100k points, no distortion | 224 µs |
| `project` 100k points, Brown-Conrady | 224 µs |
| `colormap_into` over 1M values (Turbo) | 2.6 ms |

The stated target is under 1 ms of CPU to pack 100k vertices, with one submit
per frame. Distortion costs nothing measurable because it is applied
unconditionally: undistorted intrinsics carry all-zero coefficients, for which
the polynomial is the identity, so there is no branch in the hot path.

Two tests hold the line mechanically rather than by review: a counting global
allocator asserts **zero heap allocations** between the first draw call and the
last, and `trybuild` asserts that each of the type guarantees is still a
compile error.

## Python

The same free functions, mirrored. numpy arrays cross zero-copy, and you still
own the loop.

```python
import numpy as np, fathom

r = fathom.Renderer.window("fathom", 1280, 720)
orbit = fathom.Orbit(2.0)

while r.poll():
    r.begin_frame()
    r.scene(orbit.camera(r.aspect))
    fathom.draw_grid(r, 20, 0.1, (45, 45, 55, 255))
    fathom.draw_line_strip_3d_vc(r, path, fathom.colormap(scores, 0, 1, "viridis"))
    r.end_scene()
    r.end_frame()
```

`pip install fathom`, or build from source with
`maturin develop -m crates/fathom-py/Cargo.toml`.

## Crates

| Crate | What it is |
|---|---|
| `fathom` | The umbrella. Depend on this. |
| `fathom-core` | Types only. What adapter crates depend on, and nothing else. |
| `fathom-geom` | Pure maths: pinhole projection, homography, colormaps. No GPU. |
| `fathom-render` | The wgpu backend and every `draw_*`. |
| `fathom-media` | mp4 export. Spawns `ffmpeg`; links no C libraries. |
| `fathom-py` | The Python extension. |

## License

MIT OR Apache-2.0.
