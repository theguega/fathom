//! Camera calibration, validated once at construction.
//!
//! Every type here checks its numbers in the constructor and is infallible
//! afterwards, which is what removes the `unwrap` from the projection path:
//! `project` cannot fail on bad calibration because the type is the proof.

use glam::{Mat3, Mat4, Vec3};
use thiserror::Error;

/// Why a calibration could not be built from the numbers given.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum CalibError {
    /// A focal length was zero, negative, or not finite.
    #[error("focal length must be finite and positive")]
    BadFocalLength,
    /// A principal point coordinate was not finite.
    #[error("principal point must be finite")]
    BadPrincipalPoint,
    /// A distortion coefficient was not finite.
    #[error("distortion coefficients must be finite")]
    BadDistortion,
    /// The matrix is singular, so it has no inverse and cannot be un-applied.
    #[error("matrix is singular")]
    Singular,
    /// Building a homography needs at least four non-degenerate correspondences.
    #[error("need at least 4 correspondences, got {0}")]
    TooFewCorrespondences(usize),
}

/// Pinhole intrinsics with optional Brown-Conrady distortion.
///
/// Distortion matters concretely: on a 120° wrist lens an overlay that ignores
/// it lands correctly at the image centre and tens of pixels off in the corner,
/// which sends you debugging a policy when the bug is in the viewer.
///
/// ```
/// use fathom_core::Intrinsics;
///
/// let k = Intrinsics::new(600.0, 600.0, 320.0, 240.0)?
///     .with_brown_conrady([-0.28, 0.07, 0.0], [0.0, 0.0])?;
/// assert_eq!(k.fx(), 600.0);
/// # Ok::<_, fathom_core::CalibError>(())
/// ```
///
/// # Errors
///
/// [`CalibError::BadFocalLength`] if `fx` or `fy` is not finite and positive,
/// [`CalibError::BadPrincipalPoint`] if `cx` or `cy` is not finite.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Intrinsics {
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    /// Radial `k1, k2, k3` then tangential `p1, p2`.
    dist: [f32; 5],
}

impl Intrinsics {
    /// Validate and build intrinsics from a pinhole camera matrix.
    ///
    /// # Errors
    ///
    /// See the type-level documentation.
    pub fn new(fx: f32, fy: f32, cx: f32, cy: f32) -> Result<Self, CalibError> {
        if !fx.is_finite() || !fy.is_finite() || fx <= 0.0 || fy <= 0.0 {
            return Err(CalibError::BadFocalLength);
        }
        if !cx.is_finite() || !cy.is_finite() {
            return Err(CalibError::BadPrincipalPoint);
        }
        Ok(Self {
            fx,
            fy,
            cx,
            cy,
            dist: [0.0; 5],
        })
    }

    /// Attach Brown-Conrady radial `[k1, k2, k3]` and tangential `[p1, p2]` terms.
    ///
    /// # Errors
    ///
    /// [`CalibError::BadDistortion`] if any coefficient is not finite.
    pub fn with_brown_conrady(
        mut self,
        radial: [f32; 3],
        tangential: [f32; 2],
    ) -> Result<Self, CalibError> {
        if !radial.iter().chain(&tangential).all(|c| c.is_finite()) {
            return Err(CalibError::BadDistortion);
        }
        self.dist = [
            radial[0],
            radial[1],
            radial[2],
            tangential[0],
            tangential[1],
        ];
        Ok(self)
    }

    /// Horizontal focal length in pixels.
    #[inline]
    #[must_use]
    pub const fn fx(self) -> f32 {
        self.fx
    }
    /// Vertical focal length in pixels.
    #[inline]
    #[must_use]
    pub const fn fy(self) -> f32 {
        self.fy
    }
    /// Principal point x in pixels.
    #[inline]
    #[must_use]
    pub const fn cx(self) -> f32 {
        self.cx
    }
    /// Principal point y in pixels.
    #[inline]
    #[must_use]
    pub const fn cy(self) -> f32 {
        self.cy
    }
    /// Radial `k1, k2, k3` then tangential `p1, p2`; all zero when undistorted.
    #[inline]
    #[must_use]
    pub const fn distortion(self) -> [f32; 5] {
        self.dist
    }
}

/// The rigid transform placing a camera in the world.
///
/// Both directions are computed once at construction, so neither `project` nor
/// `unproject` inverts anything per point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Extrinsics {
    world_to_camera: Mat4,
    camera_to_world: Mat4,
}

impl Extrinsics {
    /// A camera at the world origin looking down +Z.
    pub const IDENTITY: Self = Self {
        world_to_camera: Mat4::IDENTITY,
        camera_to_world: Mat4::IDENTITY,
    };

    /// Build from the world-to-camera transform, the convention OpenCV stores.
    ///
    /// # Errors
    ///
    /// [`CalibError::Singular`] if the matrix cannot be inverted.
    pub fn from_world_to_camera(m: Mat4) -> Result<Self, CalibError> {
        if m.determinant().abs() < 1e-12 {
            return Err(CalibError::Singular);
        }
        Ok(Self {
            world_to_camera: m,
            camera_to_world: m.inverse(),
        })
    }

    /// Build from the camera pose in the world, the convention a TF tree stores.
    ///
    /// # Errors
    ///
    /// [`CalibError::Singular`] if the matrix cannot be inverted.
    pub fn from_camera_to_world(m: Mat4) -> Result<Self, CalibError> {
        if m.determinant().abs() < 1e-12 {
            return Err(CalibError::Singular);
        }
        Ok(Self {
            world_to_camera: m.inverse(),
            camera_to_world: m,
        })
    }

    /// The world-to-camera transform.
    #[inline]
    #[must_use]
    pub const fn world_to_camera(self) -> Mat4 {
        self.world_to_camera
    }

    /// The camera-to-world transform.
    #[inline]
    #[must_use]
    pub const fn camera_to_world(self) -> Mat4 {
        self.camera_to_world
    }
}

/// A planar projective transform, plane pixels to image pixels.
///
/// This is the uncalibrated path: four clicked correspondences on a fixed
/// overhead view, no intrinsics needed. It is a separate type from
/// [`Intrinsics`] on purpose, so "both" and "neither" cannot be spelled.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Homography {
    forward: Mat3,
    inverse: Mat3,
}

impl Homography {
    /// The identity mapping.
    pub const IDENTITY: Self = Self {
        forward: Mat3::IDENTITY,
        inverse: Mat3::IDENTITY,
    };

    /// Build from a column-major 3x3 matrix, inverting once.
    ///
    /// # Errors
    ///
    /// [`CalibError::Singular`] if the matrix cannot be inverted.
    pub fn new(m: Mat3) -> Result<Self, CalibError> {
        let det = m.determinant();
        if !det.is_finite() || det.abs() < 1e-12 {
            return Err(CalibError::Singular);
        }
        Ok(Self {
            forward: m,
            inverse: m.inverse(),
        })
    }

    /// The plane-to-image matrix.
    #[inline]
    #[must_use]
    pub const fn forward(self) -> Mat3 {
        self.forward
    }

    /// The image-to-plane matrix.
    #[inline]
    #[must_use]
    pub const fn inverse(self) -> Mat3 {
        self.inverse
    }
}

impl Default for Homography {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Default for Extrinsics {
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// A camera looking at a target, as a ready-made [`Extrinsics`].
///
/// Convenience for examples and quick overlays; a calibrated rig uses
/// [`Extrinsics::from_world_to_camera`] with numbers from OpenCV or Kalibr.
///
/// # Errors
///
/// [`CalibError::Singular`] if `eye` and `target` coincide, or `up` is parallel
/// to the view direction.
pub fn look_at(eye: Vec3, target: Vec3, up: Vec3) -> Result<Extrinsics, CalibError> {
    let fwd = target - eye;
    if fwd.length_squared() < 1e-20
        || fwd.normalize().cross(up.normalize()).length_squared() < 1e-12
    {
        return Err(CalibError::Singular);
    }
    Extrinsics::from_world_to_camera(Mat4::look_at_rh(eye, target, up))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intrinsics_validate_once() {
        assert_eq!(
            Intrinsics::new(0.0, 1.0, 0.0, 0.0),
            Err(CalibError::BadFocalLength)
        );
        assert_eq!(
            Intrinsics::new(f32::NAN, 1.0, 0.0, 0.0),
            Err(CalibError::BadFocalLength)
        );
        let k = Intrinsics::new(600.0, 600.0, 320.0, 240.0);
        assert!(k.is_ok());
        assert_eq!(
            k.and_then(|k| k.with_brown_conrady([f32::INFINITY, 0.0, 0.0], [0.0, 0.0])),
            Err(CalibError::BadDistortion)
        );
    }

    #[test]
    fn extrinsics_precompute_both_directions() {
        let e = look_at(Vec3::new(0.0, 0.0, -2.0), Vec3::ZERO, Vec3::Y);
        let e = e.unwrap_or(Extrinsics::IDENTITY);
        let round = e.camera_to_world() * e.world_to_camera();
        assert!(
            (round - Mat4::IDENTITY)
                .to_cols_array()
                .iter()
                .all(|v| v.abs() < 1e-5)
        );
        assert_eq!(Homography::new(Mat3::ZERO), Err(CalibError::Singular));
    }
}
