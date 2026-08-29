# The primitive set

The complete public drawing surface. Deliberately short.

## Frame control

`Ctx::new` is fallible once, at the boundary. After that, drawing cannot fail.

```rust,no_run
# use fathom::prelude::*;
# fn demo(ctx: &mut Ctx, cam: &Camera) {
let mut f = begin_frame(ctx);      // draw_* take &mut Frame, so
                                   //   drawing outside a frame does not compile
let mut s = f.scene(cam);          // 3D needs a bound camera, enforced by type
draw_line_3d(&mut s, WorldPoint::ORIGIN, WorldPoint::new(1.0, 0.0, 0.0), Color::GREEN);
s.end();
f.end();                           // consumes self: no double-end, no forgotten end
# }
```

`f.scene_in(cam, rect)` confines a scene to a panel, which is how a 3D view sits
beside a video without covering it.

## Images, video, depth

Textures are allocated from a `&Ctx`, never from a `&mut Frame`, so allocating
mid-frame is a compile error rather than a stall you profile later.

| | |
|---|---|
| `upload_texture(ctx, data, w, h, fmt, filter)` | one GPU allocation |
| `update_texture(tex, data)` | non-blocking staging write, every frame |
| `draw_texture(f, tex, dst, tint)` | one quad |

## 2D, image and screen space

`draw_line_2d`, `draw_line_strip_2d`, `draw_bbox`, `draw_polygon`,
`draw_text_at`, `text_width`.

`draw_line_strip_2d` is the entire plotting subsystem: a time series is one call
per channel, with your own axis scaling.

## 3D, world space

`draw_line_3d`, `draw_line_strip_3d`, `draw_line_strip_3d_vc`, `draw_points_3d`,
`draw_wire_ellipsoid`, `draw_grid`, `draw_frames`.

Four of these earn their place because they collapse whole feature categories
into caller code:

- **`draw_line_strip_3d_vc`**, per-vertex colour, unlocks outlier scoring,
  uncertainty bands, time gradients and per-step error magnitude. All the same
  call, with a different `Vec` behind it.
- **`colormap`** is one pure function reused for depth maps, attention heatmaps,
  similarity matrices and outlier scores. Pixel encoding, not analysis.
- **`draw_wire_ellipsoid`** covers covariance, spatial variance, uncertainty
  volumes and bounding regions.
- **`draw_frames`** takes world transforms, one per link, and draws RGB axis
  triads connected in order. It works for a 7-DOF arm, a bimanual cell or a
  vehicle sensor rig precisely because it knows nothing about any of them.

## Pure maths, no GPU

`colormap`, `colormap_into`, `project`, `unproject`, `warp`, `unwarp`,
`homography_from_correspondences`.

## Deleting invariants instead of checking them

Two parallel slices that must be the same length is a runtime invariant no type
catches. Rather than a `Result` or a silently truncating `zip`, the shape
changes:

```rust,ignore
// not this: a length mismatch is representable
fn draw_line_strip_3d_vc(s: &mut Scene, pts: &[WorldPoint], colors: &[Color]);

// this: the invariant cannot be violated
fn draw_line_strip_3d_vc(s: &mut Scene, verts: &[(WorldPoint, Color)]);
```

The caller keeps a reusable scratch `Vec` and refills it each frame, which is
what the allocate-once discipline wants anyway.
