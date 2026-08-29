# Calibration

Two paths, as two separate function pairs rather than one function with a mode
flag. A caller with a calibrated rig calls `project`; a caller eyeballing a
fixed overhead view calls `warp`. Neither pays for the other, and there is no
branch in the hot path.

## Why not one type with a mode

A struct with several `Option` fields encodes states that cannot occur:

```rust,ignore
struct Calib {
    intrinsics: Option<Intrinsics>,   // four states, two of which
    homography: Option<Homography>,   // are nonsense
}
```

Separate pairs mean "both" and "neither" cannot be spelled at all.

## The pinhole path

`Intrinsics::new` validates once, at construction, and is infallible afterwards.
That is what removes the `unwrap` from the projection path: `project` cannot
fail on bad calibration because the type is the proof. Its `Option` means only
"behind the camera", which is a real geometric outcome, not an error.

```rust
use fathom::prelude::*;

let k = Intrinsics::new(600.0, 600.0, 320.0, 240.0)?
    .with_brown_conrady([-0.28, 0.07, 0.0], [0.0, 0.0])?;
let e = Extrinsics::IDENTITY;

let px = project(WorldPoint::new(0.0, 0.0, 1.0), &k, &e);
assert_eq!(px.map(|p| p.0.x), Some(320.0));
# Ok::<_, fathom::CalibError>(())
```

Distortion is applied unconditionally: undistorted intrinsics carry all-zero
coefficients, for which the polynomial is the identity. Benchmarks show
`project` costs the same either way, which is the no-branch claim measured
rather than asserted.

It matters concretely. On a 120° wrist lens, an overlay that ignores distortion
lands correctly at the image centre and tens of pixels off in the corner, which
sends you debugging a policy when the bug is in the viewer.

**Not supported:** fisheye (Kannala-Brandt) and double-sphere. They are not a
superset of Brown-Conrady and would mean an enum branch in the hot path. If a
raw ultra-wide stream ever needs viewing, the answer is a rectification map
sampled in the shader, not iterative maths in `unproject`.

## The planar path

Four clicked correspondences, no intrinsics:

```rust
use fathom::prelude::*;

let plane = [Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0), Vec2::new(1.0, 1.0), Vec2::new(0.0, 1.0)];
let image = [
    Vec2::new(100.0, 100.0), Vec2::new(500.0, 120.0),
    Vec2::new(470.0, 400.0), Vec2::new(130.0, 380.0),
];

let h = homography_from_correspondences(&plane, &image)?;
let corner = warp(PlanePoint::new(1.0, 1.0), &h);
assert!((corner.0 - image[2]).length() < 1e-2);
# Ok::<_, fathom::CalibError>(())
```

The fit is a direct linear transform with Hartley normalization, so it is stable
on raw pixel coordinates. Four points give the exact solution; more are averaged
in a least-squares sense. Degenerate input - collinear or coincident points -
returns `CalibError::Singular` rather than a silently wrong matrix.

## Why there is no `depth_to_points`

It fails the design test while hiding in a file called `geom`: it draws nothing,
and it turns a sensor product into a different data representation. That is a
data transform, and it belongs to the adapter crate that already knows the depth
encoding, the camera model, and whether the stream is rectified - and which, for
a RealSense, has the SDK's own optimized deprojection sitting right there.

Dropping it is also what closes the distortion question:

| | Cost | Verdict |
|---|---|---|
| `project`, world to pixel | closed-form polynomial | free, ship it |
| `unproject` on hover | ~6 iterations, one pixel | free, ship it |
| `depth_to_points` | the same, times 307k pixels per frame | would eat the whole budget |
