//! World-space primitives. All of them require a bound [`Scene`].
//!
//! Note what is not here: no kinematics, no robot model, no scene graph.
//! [`draw_frames`] takes the link transforms your controller already computed,
//! because having the viewer recompute them duplicates the truth and invites
//! skew between what the controller thinks the arm is doing and what you see.

use fathom_core::{Color, Mat3, Mat4, Meters, Vec3, WorldPoint};

use crate::vertex::Vertex;
use crate::{Scene, font, vertex::Topology};

/// Push one line segment. The caller has already reserved the room.
#[inline]
fn segment(s: &mut Scene<'_, '_>, a: Vec3, b: Vec3, ca: Color, cb: Color) {
    let uv = font::white_uv();
    let (a, b) = (s.clip(a), s.clip(b));
    s.push(Vertex::new(a, uv, ca));
    s.push(Vertex::new(b, uv, cb));
}

/// Draw one line segment between two world points.
pub fn draw_line_3d(s: &mut Scene<'_, '_>, a: WorldPoint, b: WorldPoint, color: Color) {
    s.begin(Topology::Lines, 2);
    segment(s, a.0, b.0, color, color);
}

/// Draw a connected polyline through world points: a trajectory, an end-effector
/// path, a predicted action chunk.
///
/// ```no_run
/// # use fathom_render::{Scene, draw_line_strip_3d};
/// # use fathom_core::{Color, WorldPoint};
/// # fn demo(s: &mut Scene<'_, '_>, executed: &[WorldPoint], predicted: &[WorldPoint]) {
/// draw_line_strip_3d(s, executed, Color::GREEN);
/// draw_line_strip_3d(s, predicted, Color::RED.with_alpha(0.5));
/// # }
/// ```
pub fn draw_line_strip_3d(s: &mut Scene<'_, '_>, pts: &[WorldPoint], color: Color) {
    let per_batch = (s.batch_limit() / 2).max(1);
    let segments = pts.len().saturating_sub(1);
    let mut done = 0;

    while done < segments {
        let n = (segments - done).min(per_batch);
        s.begin(Topology::Lines, n * 2);
        for i in done..done + n {
            let (Some(a), Some(b)) = (pts.get(i), pts.get(i + 1)) else {
                continue;
            };
            let (a, b) = (a.0, b.0);
            segment(s, a, b, color, color);
        }
        done += n;
    }
}

/// Draw a polyline with a color per vertex.
///
/// The highest-leverage primitive in the set, because it collapses whole
/// feature categories into caller code: outlier scoring, uncertainty bands,
/// time gradients, per-step error magnitude are all this one call with a
/// different `Vec` behind it.
///
/// Position and color travel together in one slice on purpose. Two parallel
/// slices would make a length mismatch representable, and no type catches that;
/// changing the shape deletes the invariant instead of checking it.
///
/// ```no_run
/// # use fathom_render::{Scene, draw_line_strip_3d_vc};
/// # use fathom_core::{Color, ColorMap, WorldPoint};
/// # fn demo(s: &mut Scene<'_, '_>, path: &[WorldPoint], error: &[f32], scratch: &mut Vec<(WorldPoint, Color)>) {
/// // Colour a trajectory by per-step tracking error.
/// scratch.clear();
/// scratch.extend(path.iter().zip(error).map(|(&p, &e)| (p, Color::GREEN.lerp(Color::RED, e))));
/// draw_line_strip_3d_vc(s, scratch);
/// # }
/// ```
pub fn draw_line_strip_3d_vc(s: &mut Scene<'_, '_>, verts: &[(WorldPoint, Color)]) {
    let per_batch = (s.batch_limit() / 2).max(1);
    let segments = verts.len().saturating_sub(1);
    let mut done = 0;

    while done < segments {
        let n = (segments - done).min(per_batch);
        s.begin(Topology::Lines, n * 2);
        for i in done..done + n {
            let (Some(&(a, ca)), Some(&(b, cb))) = (verts.get(i), verts.get(i + 1)) else {
                continue;
            };
            segment(s, a.0, b.0, ca, cb);
        }
        done += n;
    }
}

/// Draw a point cloud as camera-facing squares of a metric size.
///
/// Points are billboarded on the CPU against the bound camera's basis, so they
/// keep their world size as you orbit. The cloud arrives already deprojected:
/// fathom only ever sees points and colors, because depth deprojection belongs
/// to the adapter that knows the sensor's encoding and has the SDK's own
/// optimized path sitting right there.
pub fn draw_points_3d(s: &mut Scene<'_, '_>, verts: &[(WorldPoint, Color)], size: Meters) {
    let uv = font::white_uv();
    let (right, up) = s.basis();
    let h = size.get() * 0.5;
    let (rx, uy) = (right * h, up * h);

    let per_batch = (s.batch_limit() / 6).max(1);
    let mut done = 0;

    while done < verts.len() {
        let n = (verts.len() - done).min(per_batch);
        s.begin(Topology::Triangles, n * 6);
        for i in done..done + n {
            let Some(&(p, color)) = verts.get(i) else {
                continue;
            };
            let c = [
                s.clip(p.0 - rx + uy),
                s.clip(p.0 + rx + uy),
                s.clip(p.0 + rx - uy),
                s.clip(p.0 - rx - uy),
            ];
            for k in [0, 1, 2, 0, 2, 3] {
                if let Some(&clip) = c.get(k) {
                    s.push(Vertex::new(clip, uv, color));
                }
            }
        }
        done += n;
    }
}

/// Number of segments per ellipsoid ring. Three rings of this many segments is
/// enough to read a covariance without becoming a mesh renderer.
const RING: usize = 24;

/// Draw a wireframe ellipsoid: covariance, spatial variance, an uncertainty
/// volume, a bounding region.
///
/// `axes` maps the unit sphere to the ellipsoid, so a covariance matrix goes in
/// as its own square root - for a 1-sigma shell, the Cholesky factor or
/// `V * sqrt(D)` from an eigendecomposition, computed in your stats crate.
pub fn draw_wire_ellipsoid(s: &mut Scene<'_, '_>, center: WorldPoint, axes: Mat3, color: Color) {
    let mut ring = |plane: usize| {
        s.begin(Topology::Lines, RING * 2);
        for i in 0..RING {
            #[allow(clippy::cast_precision_loss)]
            let angle = |k: usize| core::f32::consts::TAU * k as f32 / RING as f32;
            let unit = |t: f32| match plane {
                0 => Vec3::new(t.cos(), t.sin(), 0.0),
                1 => Vec3::new(t.cos(), 0.0, t.sin()),
                _ => Vec3::new(0.0, t.cos(), t.sin()),
            };
            let a = center.0 + axes * unit(angle(i));
            let b = center.0 + axes * unit(angle(i + 1));
            segment(s, a, b, color, color);
        }
    };
    ring(0);
    ring(1);
    ring(2);
}

/// Draw a ground grid of `slices` cells per side, centred on the origin.
///
/// The grid lies in the plane spanned by the bound camera's notion of "flat":
/// X and Z, which is the graphics convention. For a Z-up robot frame, wrap the
/// call in your own transform or draw the lines yourself - it is a `for` loop.
pub fn draw_grid(s: &mut Scene<'_, '_>, slices: u32, spacing: Meters, color: Color) {
    let n = slices.max(1);
    #[allow(clippy::cast_precision_loss)]
    let half = n as f32 * spacing.get() * 0.5;
    let lines = (n as usize + 1) * 2;
    let per_batch = (s.batch_limit() / 2).max(1);
    let mut done = 0;

    while done < lines {
        let count = (lines - done).min(per_batch);
        s.begin(Topology::Lines, count * 2);
        for i in done..done + count {
            #[allow(clippy::cast_precision_loss)]
            let t = (i / 2) as f32 * spacing.get() - half;
            let (a, b) = if i % 2 == 0 {
                (Vec3::new(t, 0.0, -half), Vec3::new(t, 0.0, half))
            } else {
                (Vec3::new(-half, 0.0, t), Vec3::new(half, 0.0, t))
            };
            segment(s, a, b, color, color);
        }
        done += count;
    }
}

/// Draw an RGB axis triad per transform, connected in order.
///
/// Takes world transforms, one per link. It works for a 7-DOF arm, a bimanual
/// cell, a vehicle sensor rig or any tree precisely because it knows nothing
/// about any of them:
///
/// ```no_run
/// # use fathom_render::{Scene, draw_frames};
/// # use fathom_core::{Mat4, Meters};
/// # fn demo(s: &mut Scene<'_, '_>, left: &[Mat4; 7], right: &[Mat4; 7]) {
/// // Straight from the control stack; no FK inside fathom.
/// draw_frames(s, left, Meters(0.05));
/// draw_frames(s, right, Meters(0.05));
/// # }
/// ```
pub fn draw_frames(s: &mut Scene<'_, '_>, transforms: &[Mat4], axis_len: Meters) {
    let len = axis_len.get();
    let links = transforms.len().saturating_sub(1);
    // Six vertices of triad per transform, plus two per connecting bone.
    let per_batch = (s.batch_limit() / 8).max(1);
    let mut done = 0;

    while done < transforms.len() {
        let n = (transforms.len() - done).min(per_batch);
        let bones = if done + n > links {
            links.saturating_sub(done)
        } else {
            n
        };
        s.begin(Topology::Lines, n * 6 + bones * 2);

        for i in done..done + n {
            let Some(m) = transforms.get(i) else { continue };
            let origin = m.w_axis.truncate();
            for (axis, color) in [
                (m.x_axis, Color::RED),
                (m.y_axis, Color::GREEN),
                (m.z_axis, Color::BLUE),
            ] {
                segment(s, origin, origin + axis.truncate() * len, color, color);
            }
        }
        for i in done..(done + bones) {
            let (Some(a), Some(b)) = (transforms.get(i), transforms.get(i + 1)) else {
                continue;
            };
            let grey = Color::rgb(110, 110, 120);
            segment(s, a.w_axis.truncate(), b.w_axis.truncate(), grey, grey);
        }
        done += n;
    }
}
