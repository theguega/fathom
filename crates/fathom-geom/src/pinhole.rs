//! The calibrated pinhole path: world to pixel and back.
//!
//! Separate from the [homography](crate::warp) path on purpose. A caller with a
//! calibrated rig calls [`project`]; a caller eyeballing a fixed overhead view
//! calls [`warp`](crate::warp). Neither pays for the other, and there is no
//! mode flag to get wrong.

use fathom_core::{Extrinsics, Image, ImagePoint, Intrinsics, Meters, Point, Vec2, Vec3, World};

/// Apply Brown-Conrady distortion to a normalized image-plane point.
///
/// Unconditional: undistorted intrinsics carry all-zero coefficients, for which
/// the polynomial is the identity. That keeps the branch out of the hot path.
#[inline]
fn distort(p: Vec2, d: [f32; 5]) -> Vec2 {
    let [k1, k2, k3, p1, p2] = d;
    let r2 = p.length_squared();
    let radial = k3.mul_add(r2 * r2 * r2, k2.mul_add(r2 * r2, k1.mul_add(r2, 1.0)));
    let xy = p.x * p.y;
    Vec2::new(
        (2.0 * p1).mul_add(xy, p.x * radial) + p2 * 2.0f32.mul_add(p.x * p.x, r2),
        p.y.mul_add(radial, p1 * 2.0f32.mul_add(p.y * p.y, r2)) + 2.0 * p2 * xy,
    )
}

/// Invert [`distort`] by fixed-point iteration.
///
/// Five iterations converge to well under a tenth of a pixel for lenses up to
/// about 120°. It runs once per hovered pixel, never per point of a cloud,
/// which is exactly why fathom has no `depth_to_points`: the same maths across
/// 307k pixels would eat the whole frame budget.
#[inline]
fn undistort(p: Vec2, d: [f32; 5]) -> Vec2 {
    if d.iter().all(|c| *c == 0.0) {
        return p;
    }
    let [k1, k2, k3, p1, p2] = d;
    let mut q = p;
    for _ in 0..6 {
        let r2 = q.length_squared();
        let radial = k3.mul_add(r2 * r2 * r2, k2.mul_add(r2 * r2, k1.mul_add(r2, 1.0)));
        let xy = q.x * q.y;
        let tangential = Vec2::new(
            (2.0 * p1).mul_add(xy, p2 * 2.0f32.mul_add(q.x * q.x, r2)),
            2.0f32.mul_add(p2 * xy, p1 * 2.0f32.mul_add(q.y * q.y, r2)),
        );
        q = (p - tangential) / radial;
    }
    q
}

/// Project a world point into a calibrated camera image.
///
/// Returns `None` when the point is at or behind the image plane, which is a
/// real geometric outcome, not an error: the calibration was already validated
/// when [`Intrinsics`] was built, so this call cannot fail on bad numbers.
///
/// ```
/// use fathom_geom::{Extrinsics, Intrinsics, WorldPoint, project};
///
/// let k = Intrinsics::new(600.0, 600.0, 320.0, 240.0)?;
/// let e = Extrinsics::IDENTITY;
///
/// // A point one metre down the optical axis lands on the principal point.
/// assert_eq!(project(WorldPoint::new(0.0, 0.0, 1.0), &k, &e).map(|p| p.0.x), Some(320.0));
/// // A point behind the camera has no pixel.
/// assert!(project(WorldPoint::new(0.0, 0.0, -1.0), &k, &e).is_none());
/// # Ok::<_, fathom_core::CalibError>(())
/// ```
#[inline]
#[must_use]
#[doc(alias = "world_to_pixel")]
pub fn project(pt: Point<World>, k: &Intrinsics, e: &Extrinsics) -> Option<Point<Image>> {
    let cam = e.world_to_camera().transform_point3(pt.0);
    if cam.z <= 1e-6 {
        return None;
    }
    let n = distort(cam.truncate() / cam.z, k.distortion());
    Some(ImagePoint::new(
        k.fx().mul_add(n.x, k.cx()),
        k.fy().mul_add(n.y, k.cy()),
    ))
}

/// Lift a pixel back into the world, given the metric depth along the optical axis.
///
/// The inverse of [`project`] up to the iterative undistortion, so
/// `unproject(project(p), depth_of(p))` returns `p`. This is what turns a mouse
/// position into a metric reading, which is why it stays in the library while
/// bulk depth deprojection does not.
///
/// ```
/// use fathom_geom::{Extrinsics, Intrinsics, Meters, WorldPoint, project, unproject};
///
/// let k = Intrinsics::new(600.0, 600.0, 320.0, 240.0)?;
/// let e = Extrinsics::IDENTITY;
/// let world = WorldPoint::new(0.3, -0.2, 2.0);
///
/// let px = project(world, &k, &e).ok_or("behind the camera")?;
/// let back = unproject(px, Meters(2.0), &k, &e);
/// assert!((back.0 - world.0).length() < 1e-4);
/// # Ok::<_, Box<dyn std::error::Error>>(())
/// ```
#[inline]
#[must_use]
#[doc(alias = "pixel_to_world")]
#[doc(alias = "deproject")]
pub fn unproject(px: Point<Image>, depth: Meters, k: &Intrinsics, e: &Extrinsics) -> Point<World> {
    let n = undistort(
        Vec2::new((px.0.x - k.cx()) / k.fx(), (px.0.y - k.cy()) / k.fy()),
        k.distortion(),
    );
    let cam = Vec3::new(n.x, n.y, 1.0) * depth.get();
    Point::from_repr(e.camera_to_world().transform_point3(cam))
}

#[cfg(test)]
mod tests {
    use fathom_core::{Mat4, WorldPoint, look_at};

    use super::*;

    fn k() -> Intrinsics {
        Intrinsics::new(600.0, 600.0, 320.0, 240.0).unwrap()
    }

    #[test]
    fn round_trips_through_a_moved_camera() {
        let e = look_at(Vec3::new(1.0, 2.0, -3.0), Vec3::ZERO, Vec3::Y).unwrap();
        let world = WorldPoint::new(0.1, 0.05, 0.2);
        let px = project(world, &k(), &e).unwrap();
        let depth = Meters(e.world_to_camera().transform_point3(world.0).z);
        assert!((unproject(px, depth, &k(), &e).0 - world.0).length() < 1e-4);
    }

    #[test]
    fn distortion_moves_the_corner_and_leaves_the_centre() {
        let plain = k();
        let wide = plain
            .with_brown_conrady([-0.28, 0.07, 0.0], [0.0, 0.0])
            .unwrap();
        let e = Extrinsics::IDENTITY;

        let centre = WorldPoint::new(0.0, 0.0, 1.0);
        assert_eq!(project(centre, &plain, &e), project(centre, &wide, &e));

        let corner = WorldPoint::new(0.5, 0.4, 1.0);
        let a = project(corner, &plain, &e).unwrap().0;
        let b = project(corner, &wide, &e).unwrap().0;
        assert!(
            (a - b).length() > 10.0,
            "distortion should move a corner tens of pixels"
        );
    }

    #[test]
    fn undistort_inverts_distort() {
        let d = [-0.28, 0.07, 0.001, 0.0005, -0.0002];
        for p in [Vec2::ZERO, Vec2::new(0.4, 0.3), Vec2::new(-0.5, 0.45)] {
            assert!((undistort(distort(p, d), d) - p).length() < 1e-4);
        }
    }

    #[test]
    fn behind_and_on_the_plane_have_no_pixel() {
        let e = Extrinsics::from_world_to_camera(Mat4::IDENTITY).unwrap();
        assert!(project(WorldPoint::new(0.0, 0.0, 0.0), &k(), &e).is_none());
        assert!(project(WorldPoint::new(0.0, 0.0, -0.1), &k(), &e).is_none());
    }
}
