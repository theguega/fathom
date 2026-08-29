//! Cameras for world-space drawing.
//!
//! A [`Camera`] is a pair of matrices and nothing else: no controller, no input
//! handling, no state the library keeps for you. [`Orbit`] is the one piece of
//! convenience, because every debug session wants the same drag-to-rotate and
//! it is twenty lines the caller would otherwise write every time.

use fathom_core::{Extrinsics, Intrinsics, Meters, Radians};
use glam::{Mat4, Vec3};

/// A bound viewpoint: view and projection, already multiplied.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    view_proj: Mat4,
    right: Vec3,
    up: Vec3,
    eye: Vec3,
}

impl Camera {
    /// A perspective camera looking from `eye` at `target`.
    ///
    /// ```
    /// use fathom_render::Camera;
    /// use fathom_core::{Meters, Radians, Vec3};
    ///
    /// let cam = Camera::perspective(
    ///     Vec3::new(2.0, 2.0, 2.0), Vec3::ZERO, Vec3::Y,
    ///     Radians(1.0), 16.0 / 9.0, Meters(0.01), Meters(100.0),
    /// );
    /// // The origin sits dead centre of the image.
    /// let clip = cam.view_proj() * Vec3::ZERO.extend(1.0);
    /// assert!((clip.x / clip.w).abs() < 1e-5);
    /// ```
    #[must_use]
    pub fn perspective(
        eye: Vec3,
        target: Vec3,
        up: Vec3,
        fov_y: Radians,
        aspect: f32,
        near: Meters,
        far: Meters,
    ) -> Self {
        let view = Mat4::look_at_rh(eye, target, up);
        let proj = Mat4::perspective_rh(fov_y.get(), aspect.max(1e-3), near.get(), far.get());
        Self::from_view_proj(view, proj, eye)
    }

    /// A camera matching a calibrated real one, for overlaying world geometry
    /// on its image.
    ///
    /// Takes the same [`Intrinsics`] and [`Extrinsics`] that
    /// `fathom_geom::project` takes, so an overlay drawn in 3D and a
    /// point projected in 2D land on the same pixel. Lens distortion is *not*
    /// applied - a projection matrix cannot express it - so on a wide lens the
    /// corners drift; project the points yourself when that matters.
    #[must_use]
    pub fn from_calibration(
        k: &Intrinsics,
        e: &Extrinsics,
        width: u32,
        height: u32,
        near: Meters,
        far: Meters,
    ) -> Self {
        #[allow(clippy::cast_precision_loss)]
        let (w, h) = (width.max(1) as f32, height.max(1) as f32);
        let (n, f) = (near.get(), far.get());

        // The pinhole frustum, written straight out: +X right, +Y down, +Z
        // forward, mapped to wgpu clip space with z in 0..1.
        let proj = Mat4::from_cols_array(&[
            2.0 * k.fx() / w,
            0.0,
            0.0,
            0.0,
            0.0,
            -2.0 * k.fy() / h,
            0.0,
            0.0,
            1.0 - 2.0 * k.cx() / w,
            2.0 * k.cy() / h - 1.0,
            f / (f - n),
            1.0,
            0.0,
            0.0,
            -f * n / (f - n),
            0.0,
        ]);
        let view = e.world_to_camera();
        let eye = e.camera_to_world().w_axis.truncate();
        Self::from_view_proj(view, proj, eye)
    }

    fn from_view_proj(view: Mat4, proj: Mat4, eye: Vec3) -> Self {
        // The view matrix' rows are the camera basis in world space, which is
        // what billboarded points need to face the viewer.
        Self {
            view_proj: proj * view,
            right: Vec3::new(view.x_axis.x, view.y_axis.x, view.z_axis.x),
            up: Vec3::new(view.x_axis.y, view.y_axis.y, view.z_axis.y),
            eye,
        }
    }

    /// The combined transform, world to clip.
    #[inline]
    #[must_use]
    pub const fn view_proj(&self) -> Mat4 {
        self.view_proj
    }

    /// The camera position in world space.
    #[inline]
    #[must_use]
    pub const fn eye(&self) -> Vec3 {
        self.eye
    }

    #[inline]
    pub(crate) const fn right(&self) -> Vec3 {
        self.right
    }

    #[inline]
    pub(crate) const fn up(&self) -> Vec3 {
        self.up
    }
}

/// A turntable viewpoint, the one every debug session ends up wanting.
///
/// Plain public state: drive it from your own input handling, or set the fields
/// directly. It computes a [`Camera`]; it does not read the mouse for you.
///
/// ```
/// use fathom_render::Orbit;
/// use fathom_core::{Meters, Radians};
///
/// let mut orbit = Orbit::new(Meters(2.0));
/// orbit.rotate(0.01, 0.0);   // a drag of one hundredth of a radian
/// orbit.zoom(-1.0);          // one wheel notch closer
/// let cam = orbit.camera(16.0 / 9.0);
/// assert!(orbit.distance.get() < 2.0);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct Orbit {
    /// The point the camera looks at and rotates around.
    pub target: Vec3,
    /// Distance from `target`, clamped away from zero on [`Orbit::zoom`].
    pub distance: Meters,
    /// Rotation about the world up axis.
    pub yaw: Radians,
    /// Elevation, clamped just short of the poles so the view never flips.
    pub pitch: Radians,
    /// Vertical field of view.
    pub fov_y: Radians,
    /// World up. `Vec3::Y` for graphics data, `Vec3::Z` for most robot frames.
    pub up: Vec3,
}

impl Orbit {
    /// A turntable at `distance` from the origin, looking slightly downward.
    #[must_use]
    pub fn new(distance: Meters) -> Self {
        Self {
            target: Vec3::ZERO,
            distance,
            yaw: Radians(0.7),
            pitch: Radians(0.5),
            fov_y: Radians(core::f32::consts::FRAC_PI_4),
            up: Vec3::Y,
        }
    }

    /// Drag: yaw and pitch by radians, pitch clamped short of the poles.
    pub fn rotate(&mut self, dyaw: f32, dpitch: f32) {
        const LIMIT: f32 = core::f32::consts::FRAC_PI_2 - 0.01;
        self.yaw = Radians(self.yaw.get() + dyaw);
        self.pitch = Radians((self.pitch.get() + dpitch).clamp(-LIMIT, LIMIT));
    }

    /// Wheel: scale the distance, so each notch is a constant proportion.
    pub fn zoom(&mut self, notches: f32) {
        self.distance = Meters((self.distance.get() * 1.1f32.powf(notches)).clamp(1e-3, 1e6));
    }

    /// Middle-drag: slide the target across the view plane, scaled by distance
    /// so the world tracks the cursor at any zoom.
    pub fn pan(&mut self, dx: f32, dy: f32) {
        let cam = self.camera(1.0);
        self.target += (cam.right() * -dx + cam.up() * dy) * self.distance.get();
    }

    /// The eye position implied by the current yaw, pitch and distance.
    #[must_use]
    pub fn eye(&self) -> Vec3 {
        let (sy, cy) = self.yaw.get().sin_cos();
        let (sp, cp) = self.pitch.get().sin_cos();
        // Build the offset in a frame whose vertical axis is `up`, so the same
        // controller works for Y-up graphics data and Z-up robot frames.
        let up = self.up.normalize_or(Vec3::Y);
        let a = if up.abs().dot(Vec3::X) < 0.9 {
            Vec3::X
        } else {
            Vec3::Y
        };
        let right = up.cross(a).normalize();
        let fwd = right.cross(up);
        self.target + (right * (cp * sy) + fwd * (cp * cy) + up * sp) * self.distance.get()
    }

    /// The camera for this turntable at the given aspect ratio.
    #[must_use]
    pub fn camera(&self, aspect: f32) -> Camera {
        Camera::perspective(
            self.eye(),
            self.target,
            self.up,
            self.fov_y,
            aspect,
            Meters(self.distance.get() * 1e-3),
            Meters(self.distance.get() * 1e3),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orbit_stays_at_its_distance_and_never_flips() {
        let mut orbit = Orbit::new(Meters(3.0));
        for _ in 0..100 {
            orbit.rotate(0.3, 0.3);
            assert!((orbit.eye() - orbit.target).length() - 3.0 < 1e-3);
            assert!(orbit.pitch.get().abs() < core::f32::consts::FRAC_PI_2);
        }
        orbit.zoom(10.0);
        assert!(orbit.distance.get() > 3.0);
        orbit.zoom(-20.0);
        assert!(orbit.distance.get() < 3.0);
    }

    #[test]
    fn calibrated_camera_agrees_with_project() {
        let k = Intrinsics::new(600.0, 600.0, 320.0, 240.0).unwrap();
        let e = fathom_core::look_at(Vec3::new(0.5, -0.3, -2.0), Vec3::ZERO, Vec3::Y).unwrap();
        let cam = Camera::from_calibration(&k, &e, 640, 480, Meters(0.01), Meters(100.0));

        for world in [
            Vec3::ZERO,
            Vec3::new(0.2, 0.1, 0.05),
            Vec3::new(-0.3, 0.2, 0.1),
        ] {
            let px =
                fathom_geom::project(fathom_core::WorldPoint::from_repr(world), &k, &e).unwrap();
            let clip = cam.view_proj() * world.extend(1.0);
            let ndc = clip.truncate() / clip.w;
            // Same pixel, reached two different ways.
            let sx = (ndc.x + 1.0) * (640.0 / 2.0);
            let sy = (1.0 - ndc.y) * (480.0 / 2.0);
            assert!((sx - px.0.x).abs() < 0.05, "x: {sx} vs {}", px.0.x);
            assert!((sy - px.0.y).abs() < 0.05, "y: {sy} vs {}", px.0.y);
        }
    }
}
