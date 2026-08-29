<div align="center">

# fathom

**A low-level, zero-overhead visualization library for multimodal spatiotemporal data.**

Video, images, depth, point clouds, trajectories, kinematics, text and time series —
one set of composable drawing primitives, no framework.

[![ci](https://github.com/theguega/fathom/actions/workflows/ci.yml/badge.svg)](https://github.com/theguega/fathom/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/fathom.svg)](https://crates.io/crates/fathom)
[![docs.rs](https://img.shields.io/docsrs/fathom)](https://docs.rs/fathom)
[![license](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](#license)

</div>

---

fathom operates on **visualization only**. Pixels, points and transforms in, pixels
out. Data ingest — MCAP, LeRobot, ROS bags, video containers — lives in separate
adapter crates that depend on `fathom-core` types alone.

Built for VLA and robot-policy debugging, autonomous-driving logs, world models:
any pipeline where you need to *see* what a model did across time and space, at
the rate it actually did it.

```rust
use fathom::prelude::*;

let mut f = begin_frame(ctx);
let [left, right] = f.viewport().split_h();

// 2D: the camera stream, letterboxed rather than stretched.
draw_texture(&mut f, camera, left.fit_aspect(camera.aspect()), Color::WHITE);
draw_text_at(&mut f, ImagePoint::new(8.0, 8.0), "episode 41", TextScale::X2, Color::WHITE);

// 3D: the same instant, in world space, in the other half of the window.
let mut s = f.scene_in(&orbit.camera(right.w / right.h), right);
draw_grid(&mut s, 20, Meters(0.1), Color::GRAY);
draw_line_strip_3d(&mut s, &ee_path, Color::GREEN);
draw_frames(&mut s, &link_transforms, Meters(0.05));   // straight from your controller
s.end();

f.end();
```

## Why not just use matplotlib

Because the interesting failures are the ones you can only see *moving*.

Here is the **same 32-line multi-panel debugger** — a camera stream, a 3D
end-effector path, and a scrolling joint plot — written twice. Same panels, same
data, same update every frame. Both files are in this repo, side by side:
[`debugger_matplotlib.py`](python/benches/debugger_matplotlib.py) and
[`debugger_fathom.py`](python/benches/debugger_fathom.py).

| | lines | per frame | headroom left in a 60 Hz budget |
|---|---:|---:|---:|
| matplotlib (Agg, `set_data` fast path) | 32 | 13.9 ms | 2.7 ms |
| **fathom** | 32 | **0.21 ms** | **16.4 ms** |

**Identical code length. 65× the frame rate.**

That headroom column is the point. The 16.4 ms fathom leaves you is where your
policy inference, your video decode and your metric computation actually go.

It gets more lopsided as the data grows — one 3D trajectory plus a time series,
redrawn per frame at 1280×720:

| points | matplotlib | plotly (serialize only) | **fathom** |
|---:|---:|---:|---:|
| 1,000 | 13.3 ms | 1.1 ms | **0.18 ms** |
| 10,000 | 15.5 ms | 1.3 ms | **0.22 ms** |
| 100,000 | 46.8 ms | 6.7 ms | **0.84 ms** |

At 100k points matplotlib is down to 21 fps with nothing else on screen. The
plotly column is **serialization alone** — the JSON a browser-based viewer must
receive before it draws anything; the browser's own render is on top of that and
is not counted.

Cold start, import to first frame on screen: **fathom 18 ms, matplotlib 167 ms.**

<details>
<summary>Methodology, and where these numbers came from</summary>

Apple M-series laptop, `python/benches/compare.py`, 30–40 frame steady-state
average after a warm-up frame. Every stack gets its *fast* path, not a strawman:
matplotlib redraws with `set_data`/`set_data_3d` on existing artists and an Agg
canvas rather than rebuilding a figure; plotly is timed on `to_json` only.
fathom is measured headless, excluding the GPU readback a windowed app does not
do — including it, the 100k row is 2.55 ms rather than 0.84 ms.

Build the extension with `--release`. A debug build is ~24× slower and makes the
comparison meaningless. Reproduce with:

```sh
pip install matplotlib plotly numpy
maturin develop --release -m crates/fathom-py/Cargo.toml
python python/benches/compare.py
```

</details>

### When you should not use fathom

Honest answer, because the comparison above is narrow:

- **A static plot for a paper.** matplotlib is four lines and has ticks, legends,
  LaTeX labels and a PDF backend. fathom has none of those and will not get them.
- **Anything needing axis autoscaling, tick marks or a legend.** You would write
  them yourself. That is a real cost, and for a one-off figure it is not worth it.
- **Sharing a result in a notebook or a browser tab.** Plotly and rerun are built
  for that; fathom opens a native window or writes an mp4.

fathom is for the loop you run for hours while debugging a policy, not the figure
you paste into a slide.

## The design

Every primitive is a **free function over plain slices**. You own all state and
you own the loop; the library keeps no registry and never calls back into your
code. Multi-panel layout is `Rect` maths in your code, analysis is your stats
crate, and a heatmap is a texture.

**Two pipelines carry everything.** Textured quads and lines. Video is a quad, a
depth heatmap is a quad, text is quads, a point cloud is quads, a trajectory is
lines. Adding a primitive means writing a lowering function, never a new shader
path. The vertex arena is sized once at startup and never reallocates.

**Coordinate frames are types.** `Point<World>`, `Point<Camera>`, `Point<Image>`
and `Point<Plane>` are `#[repr(transparent)]` over `Vec2`/`Vec3` with a
zero-sized tag, so an extrinsics mix-up is a compile error rather than an overlay
that is subtly wrong in a way you notice three days later.

```rust
fn takes_world(_: WorldPoint) {}
takes_world(CameraPoint::new(1.0, 2.0, 3.0));   // does not compile
```

**The protocol is the type.** Drawing outside a frame, drawing 3D without a bound
camera, ending a frame twice, allocating a texture mid-frame, and writing to a
finished encoder are all compile errors. Each has a `trybuild` case asserting it
stays that way.

### The budget, as a number

| | |
|---|---|
| Pack 100k vertices (`draw_line_strip_3d`) | 350 µs |
| Pack 100k vertices (`draw_points_3d`, billboarded) | 389 µs |
| Pack 1k glyphs of text | 53 µs |
| `project` 100k points, no distortion | 224 µs |
| `project` 100k points, Brown-Conrady | 224 µs |
| `colormap_into` over 1M values (Turbo) | 2.6 ms |

The target is under 1 ms of CPU to pack 100k vertices, one submit per frame.
Distortion costs nothing measurable because it is applied unconditionally:
undistorted intrinsics carry all-zero coefficients, for which the polynomial is
the identity, so there is no branch in the hot path.

Two tests hold the line mechanically rather than by review: a counting global
allocator asserts **zero heap allocations** between the first draw call and the
last, and `trybuild` asserts every type guarantee is still a compile error.

## Install

> **Status: pre-release.** Not yet published to crates.io or PyPI, so the
> commands below will not work until the first release lands. Build from source
> in the meantime — the badges above are wired up and will go live with it.

```toml
[dependencies]
fathom = "0.1"
```

```sh
pip install fathom
```

From source:

```sh
git clone https://github.com/theguega/fathom && cd fathom
cargo run -p fathom --example live_viewer
maturin develop --release -m crates/fathom-py/Cargo.toml   # for the Python module
```

## Python

The same free functions, mirrored. numpy arrays cross **zero-copy**, and you
still own the loop.

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

Non-contiguous input raises rather than silently copying — including
Fortran-ordered arrays, which would otherwise be silently transposed.

Building it, running the examples, and the API in more detail:
**[`python/README.md`](python/README.md)**.

## Examples

```sh
cargo run -p fathom --example live_viewer      # the reference loop, with a producer thread
cargo run -p fathom --example robot_arm        # link transforms in, axis triads out
cargo run -p fathom --example outliers         # scoring and colormapping, in caller code
cargo run -p fathom --example depth_probe      # depth as false colour, unproject on hover
cargo run -p fathom --example homography_overlay   # four clicks to a warped floor grid
cargo run -p fathom --example headless_export --features media   # same draw code, to mp4
```

`adapter_stub` is the integration contract as runnable code: what an MCAP or
LeRobot crate needs to expose is slices of `fathom-core` types, and nothing more.

## Crates

| Crate | What it is | Dependencies |
|---|---|---|
| [`fathom`](crates/fathom) | The umbrella. Depend on this. | the three below |
| [`fathom-core`](crates/fathom-core) | Types only. What adapters depend on. | `glam`, `bytemuck` |
| [`fathom-geom`](crates/fathom-geom) | Pinhole projection, homography, colormaps. No GPU. | + none |
| [`fathom-render`](crates/fathom-render) | The wgpu backend and every `draw_*`. | + `wgpu`, `pollster` |
| [`fathom-media`](crates/fathom-media) | mp4 export. Spawns `ffmpeg`; links no C. | **none** |

`fathom-core` is two dependencies deep and stays that way: it is the only thing
the adapter ecosystem agrees on, and every breaking change to it breaks every
adapter.

## Documentation

- [The book](https://theguega.github.io/fathom/) — philosophy, coordinate
  conventions, calibration, and a "port your matplotlib script" chapter
- [API docs](https://docs.rs/fathom)

## License

MIT OR Apache-2.0, at your option.
