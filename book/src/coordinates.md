# Coordinate conventions

Coordinate frames are types. `Point<F>` is `#[repr(transparent)]` over `Vec2` or
`Vec3` with a zero-sized tag, so the tag is free at runtime and a
`&[WorldPoint]` is bit-identical to a `&[Vec3]`.

| Alias | Underlying | Meaning |
|---|---|---|
| `WorldPoint` | `Vec3` | metric world frame |
| `CameraPoint` | `Vec3` | a camera's own frame: +X right, +Y down, +Z forward |
| `ImagePoint` | `Vec2` | pixels, origin at the top-left |
| `PlanePoint` | `Vec2` | a physical plane, such as a workcell floor |

This is the single highest-value type in the library, because a bimanual cell
has a lot of frames and the mix-up is silent:

```rust,compile_fail
use fathom::prelude::*;

fn takes_world(_: WorldPoint) {}
takes_world(CameraPoint::new(1.0, 2.0, 3.0)); // does not compile
```

You can add a frame of your own; the trait is public.

```rust
use fathom::prelude::*;

#[derive(Clone, Copy, Debug)]
struct Tool;
impl CoordFrame for Tool {
    type Repr = Vec3;
}

let tip = Point::<Tool>::from_repr(Vec3::new(0.0, 0.0, 0.1));
assert_eq!(tip.0.z, 0.1);
```

## Two cameras, two conventions

There are deliberately two different "camera" ideas, and they do not use the
same axes.

- **`Extrinsics`** is a *calibration* camera, in the OpenCV convention: +X
  right, +Y down, +Z along the optical axis, so a point in front of the lens has
  positive depth. This is what `project` and `unproject` speak, and what OpenCV
  or Kalibr will have given you.
- **`Camera`** is a *rendering* viewpoint, a view and a projection matrix. This
  is what you bind to a `Scene`.

`Camera::from_calibration(&k, &e, w, h, near, far)` bridges them, so a 3D
overlay and a projected 2D point land on the same pixel. Lens distortion is the
one thing it cannot carry, because a projection matrix cannot express a
polynomial; on a wide lens, project the points yourself.

## Units

`Meters` and `Radians` are the only length and angle units, as
`#[repr(transparent)]` newtypes. `Timestamp` is nanoseconds, as an `i64`, and is
never a frame index.

## Up axis

`draw_grid` lies in the XZ plane, which is the graphics convention. `Orbit` has
an explicit `up` field: set it to `Vec3::Z` for the robot convention and the
turntable works unchanged.
