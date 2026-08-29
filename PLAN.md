# fathom

**A low-level, zero-overhead visualization library for multimodal spatiotemporal data.**

Video, images, depth, point clouds, trajectories, kinematics, text, time series - one set of composable drawing primitives, no framework.

fathom operates on **visualization only**. Pixels, points and transforms in, pixels out. Data ingest (MCAP, LeRobot, ROS bags, video containers) lives in separate adapter crates that depend on `fathom-core` types alone.

Targets: VLA / robot policy debugging, autonomous driving logs, world models, any pipeline where you need to *see* what a model did across time and space.

---

## 1. Philosophy

Borrowed wholesale from raylib. Not the dependency - the design discipline.

1. **Every primitive is a free function.** `draw_trajectory_3d(&points, color)`. No builders, no objects with lifecycle, no scene graph.
2. **Plain data in.** `&[Vec3]`, `&[f32]`, `Rect`. Never "load into the system first" beyond the one unavoidable GPU upload.
3. **Caller owns all state.** Want a trajectory to persist across frames? Keep the `Vec<Vec3>` and call the draw function every frame. The library keeps no registry.
4. **Caller owns the loop.** `begin_frame()` / draw / `end_frame()` is visible and driven by you. The library never calls back into your code.
5. **Composition is your code, not our abstraction.** Multi-panel layouts are `Rect` math in your script. Analysis is your stats crate. We draw.
6. **The library ships no analysis.** Variance, similarity, outliers, clustering - these are *examples* that prove the primitive set is sufficient, not features. If an example needs more than ~20 lines of caller code, that signals a missing primitive; if it fits, shipping it would be bloat.

### The design test

For any proposed addition, ask: does it need a new **noun** in the library, or just a new **verb over slices**? Nouns are features. Verbs are primitives. Reject nouns.

### Build it like a 1990s game

Not nostalgia - a set of constraints that happen to produce the fastest, smallest thing that does the job. A sprite-era engine had a fixed VRAM budget, a fixed sprite count, one blitter, and a loop that had to finish inside a frame. Those limits are what made them simple. Adopt them on purpose.

- **Allocate at init, never during a frame.** One vertex arena, sized at startup. Draw calls append into it. Overflow flushes and continues; it never reallocates. `Frame` hands out `&mut [Vertex]`, nothing more.
- **Two pipelines, total.** Textured quads and lines. Every single primitive lowers to one of them. Video is a quad. Depth heatmap is a quad. Text is quads. Point cloud is quads or points. Trajectory is lines. Axis triads are lines. Adding a primitive means writing a lowering function, never a new shader path.
- **One texture atlas.** Font, colormap LUTs, and any icons live in a single atlas bound once per frame. No per-primitive bind, no descriptor churn.
- **State is regenerated every frame, never diffed.** Immediate mode is the whole point: there is no retained tree to invalidate, so there is no invalidation bug class.
- **Fixed budget, stated as a number.** Target: under 1ms of CPU to pack 100k vertices, one submit per frame. Criterion enforces it; a regression fails the PR.
- **Plain old data everywhere.** `Color` is `#[repr(transparent)]` over `u32`. `Vertex` is `#[repr(C)]` and `Pod`. Everything memcpys to the GPU with `bytemuck::cast_slice`, no serialization step.

The test for whether this is holding: a debug session should open a window in under 100ms and hold 60fps while drawing a video stream, two arms, and a 100k-point cloud, on integrated graphics.

---

## 2. Toolchain and standards

| Item | Choice |
|---|---|
| Rust | 1.97.1 (latest stable, Aug 2026) |
| Edition | 2024 |
| MSRV | `rust-version = "1.85"` in `Cargo.toml`, verified in CI |
| Resolver | `resolver = "3"` (edition 2024 default) |
| Math | `glam` - SIMD `Vec3`/`Mat4`, `#[repr(C)]`, zero-cost |
| GPU | `wgpu` + `winit` |
| Text | Baked bitmap atlas, no dependency (see §5) |
| Codecs | `ffmpeg-next`, encode only, behind a non-default feature |
| Bindings | `pyo3` + `maturin`, `numpy` crate for zero-copy arrays |

### Why wgpu, not raylib-rs

raylib is the API philosophy, not the backend. `raylib-rs` drags in a C build (cmake, platform toolchains) which breaks two things that matter here: docs.rs builds and portable Python wheels. `wgpu` is pure Rust, cross-platform, gives you the same immediate-mode feel behind a thin internal command buffer, and leaves a WASM path open. The public API is unchanged either way.

`ffmpeg-next` does require C libs - unavoidable for real encode support. That is exactly why it lives behind `--features media`, so the default build stays pure Rust and docs.rs plus wheels build clean. Decode is not in the library at all (see §5).

### `rust-toolchain.toml`

```toml
[toolchain]
channel = "1.97.1"
components = ["rustfmt", "clippy", "rust-src", "llvm-tools-preview"]
```

Pin exactly. Floating `stable` makes CI failures non-reproducible.

---

## 3. Workspace layout

```
fathom/
├── Cargo.toml              # workspace root, [workspace.lints], shared deps
├── rust-toolchain.toml
├── rustfmt.toml
├── clippy.toml
├── deny.toml
├── .cargo/config.toml
├── crates/
│   ├── fathom-core/        # types only: Vec3 re-export, Color, Rect, ColorMap, colormap()
│   ├── fathom-geom/        # pure math: pinhole projection, homography, colormaps
│   ├── fathom-render/      # wgpu backend + all draw_* primitives
│   ├── fathom-media/       # ffmpeg ENCODE only (export sink), feature-gated
│   ├── fathom/             # umbrella: re-exports, prelude, the crate users depend on
│   └── fathom-py/          # pyo3 cdylib
├── examples/               # rust examples (cargo run --example)
├── python/examples/
├── benches/
└── .github/workflows/
```

**Hard rule:** `fathom-core` and `fathom-geom` have **zero** rendering dependencies. They are `no_std`-friendly, pure, independently testable and benchmarkable. If projection math needs a GPU context, the design is wrong. `fathom-geom` holds camera geometry only - no kinematics, no robot model (see §5).

### Workspace `Cargo.toml`

```toml
[workspace]
members = ["crates/*"]
resolver = "3"

[workspace.package]
version      = "0.1.0"
edition      = "2024"
rust-version = "1.85"
license      = "MIT OR Apache-2.0"
repository   = "https://github.com/<you>/fathom"

[workspace.dependencies]
glam    = { version = "0.30", features = ["bytemuck"] }
wgpu    = "26"
winit   = "0.30"
bytemuck = { version = "1", features = ["derive"] }
thiserror = "2"

[workspace.lints.rust]
missing_docs                = "deny"
missing_debug_implementations = "warn"
unsafe_code                 = "deny"      # overridden per-crate where FFI is real
unreachable_pub             = "warn"

[workspace.lints.clippy]
pedantic          = { level = "warn", priority = -1 }
unwrap_used       = "deny"
expect_used       = "deny"
panic             = "deny"
indexing_slicing  = "deny"     # v[i] is an unwrap in disguise
todo              = "deny"
fn_params_excessive_bools = "deny"   # use an enum, see §4
undocumented_unsafe_blocks = "deny"
missing_panics_doc = "deny"
exhaustive_enums  = "warn"     # #[non_exhaustive] on public enums
alloc_instead_of_core = "warn"
```

Every crate then carries `[lints] workspace = true`. One place to change policy.

---

## 4. Idiomatic Rust rules

These are review-enforceable constraints, not aspirations.

**Zero-cost**
- No `Box<dyn Trait>` anywhere in a per-frame path. Generics and enums only. Dynamic dispatch is allowed in setup code, never in `draw_*`.
- No allocation inside a draw call. Draw functions take `&[T]` and write into a pre-allocated vertex arena owned by the frame. If a signature returns `Vec<T>` it is not a draw call.
- `#[inline]` on small primitives that cross a crate boundary; nothing else. Do not sprinkle it.
- Newtypes for units (`Meters`, `FrameIdx`, `Radians`) - `#[repr(transparent)]`, free at runtime, catches the class of bug that actually bites in robotics.

**API shape**
- Accept `&[T]`, not `&Vec<T>`. Accept `impl Into<Color>` only where it removes real noise.
- `#[must_use]` on every pure function.
- Iterators over slices, no callbacks. `draw_points_3d` takes a slice; if the caller has an iterator, they collect - explicitly, in their code, where they can see the cost.
- Errors: `thiserror` in libraries, never `anyhow`. `anyhow` is allowed in examples and binaries only.

**Unsafe**
- `unsafe_code = "deny"` at the workspace level. Overridden only in `fathom-media` (FFI) and the wgpu buffer-cast path.
- Every `unsafe` block carries a `// SAFETY:` comment naming the invariant. Clippy's `undocumented_unsafe_blocks` is on.
- FFI surfaces are wrapped immediately; `unsafe` never escapes a module.

**Panics**
- `unwrap`/`expect`/`panic!` denied in library crates via clippy. `indexing_slicing` denied too - use `get()` or iterators, since a bare `v[i]` is an `unwrap` in disguise.
- The escape hatch is narrow: tests, benches, examples, and build scripts. In library code, a remaining `unwrap` needs a `// INVARIANT:` comment naming why it cannot fire, reviewed like an `unsafe` block, and it should be the signal to go restructure the type instead.
- Anything fallible returns `Result` with a `thiserror` enum. Anything infallible is proven by types, not by comments.
- Document any surviving panic path under a `# Panics` rustdoc section.

### Turn bugs into compile errors

The three tools, in order of how much they buy per line spent.

**Enums delete impossible states.**
A struct with several `Option` fields encodes states that cannot occur. Calibration is the live example: `struct Calib { intrinsics: Option<Intrinsics>, homography: Option<Homography> }` has four states, two of which are nonsense. That is why §5 has two separate function pairs instead - `project` takes `&Intrinsics`, `warp` takes `&Homography`, and "both" and "neither" simply cannot be spelled.

The same rule kills boolean parameters. `draw_grid(.., filled: bool)` reads as `draw_grid(.., true)` at the call site, which is unreadable and mixes up trivially with the next `bool`. Use a two-variant enum; it costs nothing and reads at the call site. Public enums get `#[non_exhaustive]` so adding a colormap is not a breaking change.

**Newtypes delete mixed-up values.**
`#[repr(transparent)]`, free at runtime, and they catch the bugs that actually happen in a robotics cell:

```rust
Point<World> / Point<Camera> / Point<Image> / Point<Plane>   // frame confusion
Radians(f32) / Meters(f32)                                    // unit confusion
Timestamp(i64)                                                // nanos, not seconds, not a frame index
NonZeroU32                                                    // texture dims, deletes the zero-size check
```

`warp(world_point, &h)` and `project(image_point, k, e)` become compile errors rather than overlays that are subtly wrong in a way you notice three days later. This is the single highest-leverage type in the library.

**Typestates delete out-of-order calls.**
Encode the protocol in the type, so misuse does not compile and no runtime check is needed:

```rust
let ctx = Ctx::new(&window)?;              // fallible ONCE, at the boundary
let mut f = begin_frame(&mut ctx);         // draw_* take &mut Frame, so
draw_texture(&mut f, &tex, dst, WHITE);    //   drawing outside a frame does not compile
let mut s = f.scene(&camera);              // 3D needs a bound camera, enforced by type
draw_line_3d(&mut s, a, b, GREEN);         //   drawing 3D without one does not compile
s.end();
f.end();                                   // consumes self - no double-end, no forgotten end
```

`upload_texture` takes `&Ctx`, not `&mut Frame`, so allocating a texture mid-frame is a compile error rather than a stall you profile later. The export encoder gets the same treatment: `Encoder<Open>` to `Encoder<Finished>` via a consuming `finish()`, so writing after finishing is unrepresentable.

**Parse once, then stay infallible.**
This is what actually eliminates `unwrap` in practice. Validation happens at construction, at the edge:

```rust
Intrinsics::new(fx, fy, cx, cy) -> Result<Intrinsics, CalibError>   // fx > 0 checked here, once
fn project(pt: Point<World>, k: &Intrinsics, e: &Extrinsics) -> Option<Point<Image>>
```

After that, `project` cannot fail on bad calibration - the type is proof. Its `Option` means only "behind the camera," which is a real geometric outcome, not an error. Every `Result` in the library should be traceable to a genuine boundary: GPU init, file I/O, FFI, or user-supplied numbers.

**Where the type system stops, delete the invariant instead.**
Two parallel slices that must be the same length is a runtime invariant no type catches. Rather than `Result` or a silent truncating `zip`, change the shape:

```rust
// not this - a length mismatch is representable
fn draw_line_strip_3d_vc(f: &mut Frame, pts: &[Point<World>], colors: &[Color]);

// this - the invariant cannot be violated
fn draw_line_strip_3d_vc(f: &mut Frame, verts: &[(Point<World>, Color)]);
fn colormap_into(values: &[f32], range: Range<f32>, map: ColorMap, out: &mut [Color]);
```

The caller keeps a reusable `Vec<(Point<World>, Color)>` scratch buffer and refills it each frame, which is what the arcade discipline in §1 wants anyway: allocate once, reuse forever. `colormap` keeps its allocating form for convenience; `colormap_into` is the one used in a loop.

---

## 5. The primitive set

The complete public drawing surface. Deliberately short.

```rust
// --- frame control (typestate: see §4) ---
fn Ctx::new(window: &Window) -> Result<Ctx, InitError>;   // fallible once, at the boundary
fn begin_frame(ctx: &mut Ctx) -> Frame<'_>;
impl Frame<'_> {
    fn scene(&mut self, cam: &Camera) -> Scene<'_>;       // 3D requires a bound camera
    fn end(self);                                          // consumes, no double-end
}

// --- images / video / depth (Ctx, not Frame - no mid-frame allocation) ---
fn upload_texture(ctx: &Ctx, data: &[u8], w: NonZeroU32, h: NonZeroU32, fmt: Format) -> Texture;
fn update_texture(tex: &Texture, data: &[u8]);
fn draw_texture(f: &mut Frame, tex: &Texture, dst: Rect, tint: Color);

// --- 2D, image/screen space ---
fn draw_line_2d(f: &mut Frame, a: Point<Image>, b: Point<Image>, color: Color);
fn draw_line_strip_2d(f: &mut Frame, pts: &[Point<Image>], color: Color);
fn draw_bbox(f: &mut Frame, r: Rect, color: Color);
fn draw_polygon(f: &mut Frame, pts: &[Point<Image>], color: Color);
fn draw_text_at(f: &mut Frame, pos: Point<Image>, text: &str, scale: TextScale, color: Color);

// --- 3D, world space, requires a Scene ---
fn draw_line_3d(s: &mut Scene, a: Point<World>, b: Point<World>, color: Color);
fn draw_line_strip_3d(s: &mut Scene, pts: &[Point<World>], color: Color);
fn draw_line_strip_3d_vc(s: &mut Scene, verts: &[(Point<World>, Color)]);
fn draw_points_3d(s: &mut Scene, verts: &[(Point<World>, Color)], size: Meters);
fn draw_wire_ellipsoid(s: &mut Scene, center: Point<World>, axes: Mat3, color: Color);
fn draw_grid(s: &mut Scene, slices: u32, spacing: Meters, color: Color);
fn draw_frames(s: &mut Scene, transforms: &[Mat4], axis_len: Meters);

// --- pure math, no GPU, in fathom-geom ---
fn colormap(values: &[f32], range: Range<f32>, map: ColorMap) -> Vec<Color>;
fn colormap_into(values: &[f32], range: Range<f32>, map: ColorMap, out: &mut [Color]);

// calibrated pinhole path. Intrinsics carries optional Brown-Conrady k1..k3, p1, p2.
fn project(pt: Point<World>, k: &Intrinsics, e: &Extrinsics) -> Option<Point<Image>>;  // closed form
fn unproject(px: Point<Image>, depth: Meters, k: &Intrinsics, e: &Extrinsics) -> Point<World>;

// planar homography path - separate function, not a mode flag
fn warp(px: Point<Plane>, h: &Homography) -> Point<Image>;
fn unwarp(px: Point<Image>, h: &Homography) -> Point<Plane>;
fn homography_from_correspondences(src: &[Vec2], dst: &[Vec2]) -> Option<Homography>;
```

Three of these earn their place specifically because they collapse whole feature categories into caller code:

- **`draw_line_strip_3d_vc`** (per-vertex color) - unlocks outlier scoring, uncertainty bands, time gradients, per-step error magnitude. All the same call.
- **`colormap`** - one pure function reused for depth maps, attention heatmaps, similarity matrices, outlier scores. Pixel encoding, not analysis.
- **`draw_wire_ellipsoid`** - covariance, spatial variance, uncertainty volumes, bounding regions.
- **`draw_frames`** - takes world transforms, one per link, and draws RGB axis triads connected in order. Works for a 7-DOF arm, a bimanual cell, a vehicle sensor rig, or any tree, because it knows nothing about any of them.

What is deliberately **absent**: plotting subsystem (a time series is `draw_line_strip_2d` per channel), panel/layout manager (`Rect` math in your code), overlay trait objects, timeline widget, analysis crate, **kinematics**, **depth-to-point-cloud conversion**.

### No kinematics dependency

`forward_kinematics` is not in the library and fathom depends on no kinematics crate (`k`, `kinetix`, or otherwise). FK is robotics domain logic; drawing is not. Your control stack already computes link transforms in its loop - having the viz layer recompute them duplicates the truth and invites version skew between what the controller thinks the arm is doing and what the screen shows. The caller passes `&[Mat4]`, fathom draws it.

One idea worth stealing without the dependency: kinetix's type-driven frame safety. Newtype the coordinate frame (`Point<World>`, `Point<Camera>`, `Point<Image>`) so `project` can only be called with operands that agree. `#[repr(transparent)]`, free at runtime, catches the extrinsics mixup that otherwise costs an afternoon.

`project` and `unproject` stay because you cannot draw an overlay onto a camera image, or click a pixel, without them - they are rendering geometry, not sensor processing.

### No depth conversion, and therefore no distortion problem

`depth_to_points` is not in the library. It failed the design test while hiding in a file called `geom`: it draws nothing, it takes a sensor product and returns a different data representation. That is a data transform, which belongs to the adapter crate that already knows the depth encoding, the camera model, and whether the stream is rectified - and which, for a RealSense, has the SDK's own optimized deprojection sitting right there.

Dropping it is also what closes the distortion question, because the two directions have very different costs:

| | Cost | Verdict |
|---|---|---|
| `project`, world to pixel | closed-form polynomial | free, ship it |
| `unproject` on hover | ~5 Newton iterations, one pixel | free, ship it |
| `depth_to_points` | same iteration times 307k pixels per frame | would have eaten the entire 1ms budget |

So `Intrinsics` carries optional Brown-Conrady coefficients (`k1..k3, p1, p2`) and `project` applies them. That matters concretely: on a 120° wrist lens, an overlay that ignores distortion lands correctly at the image center and tens of pixels off in the corner, which sends you debugging a policy when the bug is in the viewer. Five floats and a polynomial is a cheap fix for that.

What is **not** supported: fisheye/equidistant (Kannala-Brandt) and the double-sphere models, which are not a superset of Brown-Conrady and would mean an enum branch in the hot path. If a raw ultra-wide stream ever needs viewing, the answer is a `rectify_map` texture sampled in the shader - a rendering feature costing zero CPU, on the existing textured-quad pipeline - not iterative math in `unproject`.

In practice this rarely bites: RealSense rectifies in the SDK, and any pipeline that has run Kalibr or OpenCV calibration usually stores the rectified stream.

### Text: a bitmap font, and that is all

No shaping stack, no `cosmic-text`, no `swash`, no runtime font loading. A monospace atlas baked into the binary as a `const [u8; N]`, ASCII plus Latin-1, integer-scaled so it stays crisp at 1x/2x/3x.

`draw_text_at` emits one quad per glyph into the same vertex arena and the same pipeline as `draw_texture`. That is the entire implementation - maybe 60 lines including the atlas lookup. Text costs the library nothing because text *is* sprites, exactly as it was on hardware that had no other option.

What this gives up: non-Latin instruction strings render as boxes. For a VLA debugger that reads mostly English instructions and numeric readouts, that is the right trade. Anyone who needs CJK or Arabic instruction text turns on `--features text-shaping`, which swaps in a real shaper behind the identical `draw_text_at` signature. Off by default, absent from the dependency tree entirely when unused.

### Calibration: both paths, no mode flag


Full pinhole (intrinsics + extrinsics, with optional radial-tangential distortion coefficients on `Intrinsics`) and planar homography are **two separate function pairs**, not one function with a `Projection` enum. A caller with a calibrated rig calls `project`; a caller eyeballing a fixed overhead view calls `warp`. Neither pays for the other, and there is no branch in the hot path.

The frame typing does the work that a mode flag would otherwise do badly. `Point<F>` is `#[repr(transparent)]` over `Vec2`/`Vec3` with a zero-sized frame marker, so `warp(world_point, &h)` is a compile error rather than a silently wrong overlay. Free at runtime, and it is the single highest-value type in the library given how many coordinate frames a bimanual cell has.

### Live streaming: fathom never owns a thread

Live and playback are the same code path, because the library has no concept of a "source." The caller pulls a frame from wherever it came from and calls `update_texture`. That is the entire integration surface.

Concretely, three rules:

1. **No internal threads, no internal buffering.** The adapter crate owns the decode thread and the bounded channel. `update_texture` is a non-blocking staging-buffer write; if no new frame arrived, the caller simply does not call it and the previous texture is redrawn. A slow producer degrades to a stale frame, never a stalled render loop.
2. **Timestamps, not frame indices, are the shared axis.** `Timestamp(i64)` in nanoseconds. Video arrives at 30Hz, joint states at 500Hz, a language instruction once per episode. A `frame_idx` cannot express that. Any resampling or nearest-neighbor lookup is caller code - `slice.partition_point(|s| s.t <= now)` is one line and needs no library support.
3. **Seeking does not exist in the library.** Scrubbing is the caller choosing which slice to hand to the draw calls. In live mode there is nothing to seek; in playback the adapter crate seeks. Same draw code either way, which is the fidelity test again.

### Interop: adapters are external, and depend on `fathom-core` only

MCAP, LeRobot, ROS bags, raw dumps, and whatever comes next are **separate crates outside this workspace**. fathom operates on visualization. It takes pixels, points, and transforms in, and produces pixels out.

The ecosystem contract is deliberately thin: adapter crates depend on `fathom-core` alone (a types-only crate: `Point<F>`, `Color`, `Rect`, `Timestamp`, `Intrinsics`, `Extrinsics`, `Homography`) and expose plain slices of those. No `Source` trait, no plugin registry, no dynamic dispatch. An adapter is just a crate whose functions happen to return types fathom already understands.

This is why `fathom-core` must stay tiny and stable: it is the only thing the ecosystem agrees on, and every breaking change to it breaks every adapter. It gets `cargo-semver-checks` treatment more seriously than anything else in the workspace.

Consequence: **decode leaves the library.** `fathom-media` keeps ffmpeg *encode* only, because "render N frames to an mp4" is an output of the renderer. Decode belongs in the LeRobot or MCAP adapter that already knows the container layout. A thin first-party `fathom-ffmpeg` adapter crate can ship alongside so nobody is stranded, but it lives outside the core the same as any third-party adapter.

---

## 6. Documentation as a first-class deliverable

**Enforced by the compiler**

```rust
// crates/fathom/src/lib.rs
#![doc = include_str!("../README.md")]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![deny(missing_docs)]
```

`missing_docs` is `deny`, not `warn`. An undocumented public item fails the build.

**Every public function's rustdoc has:**
- One-line summary, imperative mood.
- A runnable ```` ```rust ```` doctest. Doctests are the example suite - they cannot drift from the API, because CI compiles and runs them.
- `# Errors` for anything returning `Result`, `# Panics` if any path can panic.
- Intra-doc links (`[`colormap`]`) rather than prose references.
- `#[doc(alias = "heatmap")]` on `colormap`, and similar aliases wherever the domain word differs from the function name. This makes docs.rs search actually find things.

**Feature visibility**

```toml
[package.metadata.docs.rs]
all-features = true
rustdoc-args = ["--cfg", "docsrs", "--generate-link-to-definition"]
```

```rust
#[cfg(feature = "media")]
#[cfg_attr(docsrs, doc(cfg(feature = "media")))]
pub mod media;
```

Feature-gated items render with a visible badge on docs.rs instead of silently vanishing.

**Guide-level docs**

`mdBook` in `book/`, covering: the philosophy, the design test, coordinate conventions, calibration formats, a "port your matplotlib script" chapter. Published alongside the API docs.

**Python docs**

- PyO3 `#[pyfunction]` doc comments become real Python docstrings. Write them in the Rust source; single source of truth.
- Ship `.pyi` stubs (generated via `pyo3-stub-gen`, checked into the repo, diffed in CI) so editors and mypy work.
- `pdoc` renders the stubs plus docstrings to HTML.

**Publishing**

Both trees deploy to GitHub Pages on every push to `main`: `/` for the book, `/api/rust/` for rustdoc, `/api/python/` for pdoc. docs.rs handles released versions automatically.

---

## 7. CI/CD

### `.github/workflows/ci.yml`

```yaml
name: ci
on: [push, pull_request]

env:
  CARGO_TERM_COLOR: always
  RUSTFLAGS: -D warnings

jobs:
  fmt:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with: { components: rustfmt }
      - run: cargo fmt --all --check
      - uses: taiki-e/install-action@v2
        with: { tool: taplo-cli }
      - run: taplo fmt --check          # Cargo.toml formatting too

  clippy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with: { components: clippy }
      - uses: Swatinem/rust-cache@v2
      - run: cargo clippy --workspace --all-targets --all-features

  test:
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      - uses: taiki-e/install-action@nextest
      - run: cargo nextest run --workspace --all-features
      - run: cargo test --workspace --doc --all-features   # doctests

  msrv:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@1.85.0
      - run: cargo check --workspace --all-features

  docs:
    runs-on: ubuntu-latest
    env:
      RUSTDOCFLAGS: -D warnings -D rustdoc::broken_intra_doc_links --cfg docsrs
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@nightly
      - run: cargo doc --workspace --no-deps --all-features

  deny:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: EmbarkStudios/cargo-deny-action@v2   # licenses, advisories, bans

  semver:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: obi1kenobi/cargo-semver-checks-action@v2

  coverage:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with: { components: llvm-tools-preview }
      - uses: taiki-e/install-action@cargo-llvm-cov
      - run: cargo llvm-cov --workspace --lcov --output-path lcov.info
      - uses: codecov/codecov-action@v4

  bench:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo bench --bench primitives -- --save-baseline pr
```

`cargo-semver-checks` matters more than it looks: with a Python binding downstream, an accidental breaking change in the Rust API silently breaks wheels. Catch it at PR time.

Benchmarks (criterion) target the things where regressions actually hurt: `colormap_into` over 1M values, `project` with distortion over 100k points, and vertex packing for a 100k-point cloud against the stated 1ms budget. A separate test asserts **zero heap allocations between `begin_frame` and `end`**, using a counting global allocator - the arcade discipline from §1 enforced mechanically rather than by review.

The §4 type guarantees get the same treatment via `trybuild` compile-fail tests, so they cannot silently regress: drawing outside a frame, drawing 3D without a `Scene`, `warp`-ing a `Point<World>`, ending a `Frame` twice, and writing to a finished `Encoder` each have a `tests/ui/*.rs` case asserting the error. A guarantee that is not tested is a guarantee someone refactors away.

### `.github/workflows/release.yml`

- `release-plz` for version bumps, changelog, and `cargo publish` of all crates in dependency order.
- `maturin` build matrix: manylinux x86_64/aarch64, macOS universal2, Windows x64, CPython 3.10-3.13 plus abi3. Publish to PyPI via trusted publishing (OIDC, no API token in secrets).
- `cargo-dist` for prebuilt example binaries, optional.

### Config files

```toml
# rustfmt.toml
edition = "2024"
imports_granularity = "Crate"
group_imports = "StdExternalCrate"
```

```toml
# clippy.toml
avoid-breaking-exported-api = false   # pre-1.0
disallowed-methods = [
  { path = "std::vec::Vec::new", reason = "no allocation in draw paths" },
]
```

```toml
# .cargo/config.toml
[alias]
xtask = "run --package xtask --"
lint  = "clippy --workspace --all-targets --all-features -- -D warnings"
```

Plus a pre-commit hook running `cargo fmt`, `taplo fmt`, and `typos`.

---

## 8. Python bindings

Mirror the Rust free functions **verbatim**. No classes, no "Pythonic" wrapper. That wrapper layer is exactly where bloat re-enters.

```rust
// crates/fathom-py/src/lib.rs
use numpy::{PyReadonlyArray2, PyReadonlyArray1};
use pyo3::prelude::*;

/// Draw a 3D polyline with one color per vertex.
///
/// Args:
///     points: (N, 3) float32 array of positions.
///     colors: (N, 4) uint8 array of RGBA.
#[pyfunction]
fn draw_line_strip_3d_vc(
    frame: &mut PyFrame,
    points: PyReadonlyArray2<'_, f32>,
    colors: PyReadonlyArray2<'_, u8>,
) -> PyResult<()> {
    let pts = bytemuck::cast_slice(points.as_slice()?);   // zero-copy
    let cols = bytemuck::cast_slice(colors.as_slice()?);
    fathom::draw_line_strip_3d_vc(&mut frame.0, pts, cols);
    Ok(())
}

#[pymodule]
fn fathom(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(draw_line_strip_3d_vc, m)?)?;
    // ...
    Ok(())
}
```

Rules:
- Contiguous numpy arrays cross the boundary **zero-copy** via `bytemuck::cast_slice`. Non-contiguous input errors out loudly rather than silently copying.
- `abi3-py310` feature - one wheel per platform across all Python 3.10+.
- Release the GIL (`py.detach(|| ...)`) around any call that blocks on decode or GPU submit.
- Doc comments on the Rust `#[pyfunction]` are the Python docstrings. Never duplicate.

---

## 9. Examples

Examples exist to prove the primitive set is sufficient. If one of these needs a library change, that is the signal.

### Rust

**`examples/live_viewer.rs`** - pull frames from a channel, draw with a text HUD, timestamp-aligned joint overlay. The reference for "caller owns the loop" and for live mode: the example owns the producer thread, fathom does not.

**`examples/headless_export.rs`** - identical draw calls, different init, frames encoded to mp4. The fidelity test: if the draw code differs between this and `live_viewer`, the API leaked a display or a liveness assumption.

**`examples/outliers.rs`** - the whole "outlier analysis" story, in caller code, with no analysis crate:

```rust
// caller computes scores; fathom knows nothing about "outliers"
let scores: Vec<f32> = demos.iter()
    .map(|d| dtw_distance(&d.ee_path, &reference.ee_path))
    .collect();
colormap_into(&scores, 0.0..max, ColorMap::Turbo, &mut colors);  // scratch, no alloc

let mut f = begin_frame(&mut ctx);
let mut s = f.scene(&orbit_cam);
for (demo, &c) in demos.iter().zip(&colors) {
    draw_line_strip_3d(&mut s, &demo.ee_path, c.with_alpha(0.3));
}
draw_line_strip_3d(&mut s, &reference.ee_path, WHITE);
s.end();
f.end();
```

Swap `dtw_distance` for cosine similarity and you have the similarity example. Slice `demos` by subtask boundary first and you have the granularity example. **The examples vary, the primitives do not.**

**`examples/variance.rs`** - overplot N demos at low alpha, plus `draw_wire_ellipsoid` from a caller-computed covariance at each waypoint.

**`examples/similarity_matrix.rs`** - NxN distances to `colormap` to `upload_texture` to `draw_texture`. A heatmap is a texture. There is no heatmap primitive.

**`examples/robot_arm.rs`** - transforms straight from the control stack, no FK inside fathom:

```rust
// your controller already has these, every tick
let left:  [Mat4; 7] = robot.left.link_transforms();
let right: [Mat4; 7] = robot.right.link_transforms();

let mut f = begin_frame(&mut ctx);
let mut s = f.scene(&orbit_cam);
draw_frames(&mut s, &left,  Meters(0.05));
draw_frames(&mut s, &right, Meters(0.05));
draw_line_strip_3d(&mut s, &executed_ee, GREEN);
draw_line_strip_3d(&mut s, &predicted_chunk, RED.with_alpha(0.5));
s.end();
f.end();
```

**`examples/fk_minimal.rs`** - for callers holding only joint angles. This is why no kinematics crate is needed: a serial chain is an accumulate-and-collect, in the caller's code, with glam already on hand.

```rust
struct Link { fixed: Mat4, axis: Vec3 }

fn fk(joints: &[f32], chain: &[Link]) -> Vec<Mat4> {
    let mut t = Mat4::IDENTITY;
    joints.iter().zip(chain).map(|(&q, link)| {
        t *= link.fixed * Mat4::from_axis_angle(link.axis, q);
        t
    }).collect()
}

// feeds draw_frames directly
draw_frames(&mut s, &fk(&joint_angles, &wam_chain), Meters(0.05));
```

Anyone needing URDF parsing, IK, or dynamics reaches for `k` or `kinetix` in their own crate and passes the result in. That boundary is the point.

**`examples/depth_probe.rs`** - depth buffer to false color via `colormap_into`, drawn as a texture; `unproject` on hover to read metric depth at the cursor. The point cloud comes from the adapter already deprojected, so fathom only ever sees `&[(Point<World>, Color)]`.

**`examples/homography_overlay.rs`** - four clicked correspondences to `homography_from_correspondences`, then a workcell floor grid warped onto a fixed overhead camera view. The uncalibrated path, end to end, in under 40 lines.

**`examples/adapter_stub.rs`** - a ~30-line fake adapter emitting timestamped frames and joint states, showing exactly what an MCAP or LeRobot crate needs to expose: slices of `fathom-core` types, nothing more. This doubles as the integration contract documentation.

### Python

**`python/examples/scrub.py`** - the live viewer, ~40 lines.

**`python/examples/attention.py`** - cross-attention map to `colormap` to texture, alpha-blended over the camera frame.

**`python/examples/outliers.py`** - the Rust example above, using `scipy` for the distance metric. Same shape, same call names, proving the binding is a mirror and not a reinterpretation.

---

## 10. Build order

Sequenced so you have something usable before touching the hard parts.

**Phase 1 - runnable tool.** `fathom-core` types (including `Point<F>` and `Timestamp`), wgpu context and `Frame`, texture upload/update, 2D primitives, text, `colormap`. Ships `live_viewer` and `adapter_stub`. No ffmpeg, no 3D. Live streaming works from day one because there is nothing to add for it.

**Phase 2 - spatial.** 3D primitives, orbit camera, `draw_frames`, `fathom-geom` (both the pinhole and homography pairs), point clouds, ellipsoids. Ships `robot_arm`, `fk_minimal`, `depth_probe`, `homography_overlay`, `outliers`.

**Phase 3 - export.** Offscreen render target, `fathom-media` encode, headless mode. Ships `headless_export`.

**Phase 4 - bindings and docs.** `fathom-py`, stub generation, wheel matrix, mdBook, Pages deploy.

**Outside the workspace, in parallel:** MCAP and LeRobot adapter crates, plus `fathom-ffmpeg` for plain video files. These can be built by anyone against `fathom-core` as soon as phase 1 lands, which is the point of keeping that crate tiny.

CI, lints, and `missing_docs = deny` are in place from **day one**, not phase 4. Retrofitting documentation onto a hundred public functions is the failure mode this plan exists to avoid.

---

## Decisions

No open questions remain. Everything below is settled; each one is a deletion, which is the pattern worth noticing.

- **`fathom-core` freezes at the end of phase 2.** Late enough that the type set is proven by real primitives, early enough that adapter authors are not chasing breakage. Stricter `cargo-semver-checks` treatment than any other crate from that point on.
- **Text is a baked bitmap atlas.** Real shaping only behind an off-by-default feature.
- **Calibration supports both pinhole and homography**, as separate function pairs rather than a mode enum.
- **Distortion is Brown-Conrady only, applied in `project` only.** No fisheye, no `depth_to_points`, no iterative undistortion anywhere in a per-frame path.
- **Live streaming is the default assumption**, and costs nothing because fathom owns no threads and no buffering.
- **Ingest is external.** MCAP, LeRobot, video decode and depth deprojection live in adapter crates depending on `fathom-core` alone.
- **No kinematics.** The caller passes `&[Mat4]`.
- **No analysis.** Variance, similarity and outliers are examples, not features.
