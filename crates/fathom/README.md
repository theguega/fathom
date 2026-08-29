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

## Crates

| Crate | What it is |
|---|---|
| `fathom` | The umbrella. Depend on this. |
| `fathom-core` | Types only. What adapter crates depend on, and nothing else. |
| `fathom-geom` | Pure maths: pinhole projection, homography, colormaps. No GPU. |
| `fathom-render` | The wgpu backend and every `draw_*`. |

## License

MIT OR Apache-2.0.
