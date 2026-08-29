//! The uncalibrated planar path: plane to image and back.
//!
//! Four clicked correspondences on a fixed overhead view, no intrinsics needed.
//! Kept separate from [`project`](crate::project) so that "calibrated and
//! planar at the same time" cannot be spelled.

use fathom_core::{CalibError, Homography, Image, Mat3, Plane, Point, Vec2, Vec3};

#[inline]
fn apply(m: Mat3, p: Vec2) -> Vec2 {
    let v = m * Vec3::new(p.x, p.y, 1.0);
    // A finite homography maps a finite point to w != 0 except exactly on the
    // line at infinity; guarding here keeps the result finite instead of NaN.
    let w = if v.z.abs() < 1e-12 { 1e-12 } else { v.z };
    Vec2::new(v.x / w, v.y / w)
}

/// Map a point on the physical plane to its pixel in the image.
///
/// ```
/// use fathom_geom::{Homography, PlanePoint, warp};
///
/// let h = Homography::IDENTITY;
/// assert_eq!(warp(PlanePoint::new(3.0, 4.0), &h).0.to_array(), [3.0, 4.0]);
/// ```
#[inline]
#[must_use]
#[doc(alias = "perspective_transform")]
pub fn warp(px: Point<Plane>, h: &Homography) -> Point<Image> {
    Point::from_repr(apply(h.forward(), px.0))
}

/// Map a pixel back onto the physical plane.
///
/// ```
/// use fathom_geom::{Homography, PlanePoint, unwarp, warp};
///
/// let h = Homography::IDENTITY;
/// let p = PlanePoint::new(3.0, 4.0);
/// assert_eq!(unwarp(warp(p, &h), &h), p);
/// ```
#[inline]
#[must_use]
pub fn unwarp(px: Point<Image>, h: &Homography) -> Point<Plane> {
    Point::from_repr(apply(h.inverse(), px.0))
}

/// Hartley normalization: centre on the centroid and scale to a mean radius of
/// `sqrt(2)`. Without it the direct linear transform is badly conditioned on
/// pixel-scale coordinates.
fn normalize(pts: &[Vec2]) -> Option<(Mat3, Mat3)> {
    #[allow(clippy::cast_precision_loss)]
    let n = pts.len() as f32;
    let centroid = pts.iter().copied().sum::<Vec2>() / n;
    let mean_dist = pts.iter().map(|p| (*p - centroid).length()).sum::<f32>() / n;
    if !mean_dist.is_finite() || mean_dist < 1e-12 {
        return None;
    }
    let s = core::f32::consts::SQRT_2 / mean_dist;
    let t = Mat3::from_cols_array(&[
        s,
        0.0,
        0.0,
        0.0,
        s,
        0.0,
        -s * centroid.x,
        -s * centroid.y,
        1.0,
    ]);
    let inv = Mat3::from_cols_array(&[
        1.0 / s,
        0.0,
        0.0,
        0.0,
        1.0 / s,
        0.0,
        centroid.x,
        centroid.y,
        1.0,
    ]);
    Some((t, inv))
}

/// Gauss-Jordan with partial pivoting on the 8x8 normal equations.
///
/// Every index is bounded by a compile-time constant loop over a fixed-size
/// array, so `indexing_slicing` is allowed here: there is no runtime length to
/// get wrong.
#[allow(clippy::indexing_slicing)]
fn solve8(mut m: [[f64; 9]; 8]) -> Option<[f64; 8]> {
    for col in 0..8 {
        let pivot = (col..8).max_by(|&a, &b| {
            m[a][col]
                .abs()
                .partial_cmp(&m[b][col].abs())
                .unwrap_or(core::cmp::Ordering::Equal)
        })?;
        if m[pivot][col].abs() < 1e-12 {
            return None;
        }
        m.swap(col, pivot);
        let d = m[col][col];
        for v in &mut m[col] {
            *v /= d;
        }
        for row in 0..8 {
            if row != col {
                let f = m[row][col];
                let pivot_row = m[col];
                for (dst, src) in m[row].iter_mut().zip(&pivot_row) {
                    *dst -= f * src;
                }
            }
        }
    }
    let mut out = [0.0; 8];
    for (o, row) in out.iter_mut().zip(&m) {
        *o = row[8];
        if !o.is_finite() {
            return None;
        }
    }
    Some(out)
}

/// Fit a homography to four or more plane-to-image correspondences.
///
/// Least squares over the direct linear transform, with Hartley normalization
/// so the fit is stable on raw pixel coordinates. Four points give the exact
/// solution; more are averaged.
///
/// ```
/// use fathom_geom::{PlanePoint, Vec2, homography_from_correspondences, warp};
///
/// // A workcell floor square, as seen by a fixed overhead camera.
/// let plane = [Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0), Vec2::new(1.0, 1.0), Vec2::new(0.0, 1.0)];
/// let image = [Vec2::new(100.0, 100.0), Vec2::new(500.0, 120.0), Vec2::new(470.0, 400.0), Vec2::new(130.0, 380.0)];
///
/// let h = homography_from_correspondences(&plane, &image)?;
/// let fitted = warp(PlanePoint::new(1.0, 1.0), &h);
/// assert!((fitted.0 - image[2]).length() < 1e-2);
/// # Ok::<_, fathom_core::CalibError>(())
/// ```
///
/// # Errors
///
/// [`CalibError::TooFewCorrespondences`] with fewer than four points on either
/// side, and [`CalibError::Singular`] when the points are degenerate - all
/// collinear, or coincident - so no unique plane maps between them.
#[doc(alias = "find_homography")]
#[doc(alias = "dlt")]
#[allow(clippy::many_single_char_names)] // x, y, u, v are the names in every DLT derivation
pub fn homography_from_correspondences(
    src: &[Vec2],
    dst: &[Vec2],
) -> Result<Homography, CalibError> {
    let n = src.len().min(dst.len());
    if n < 4 {
        return Err(CalibError::TooFewCorrespondences(n));
    }
    let (ts, _) = normalize(src).ok_or(CalibError::Singular)?;
    let (td, td_inv) = normalize(dst).ok_or(CalibError::Singular)?;

    // Accumulate the normal equations A^T A and A^T b directly: 2N rows folded
    // into a fixed 8x9 block, so this allocates nothing whatever N is.
    let mut normal = [[0.0f64; 9]; 8];
    for (s, d) in src.iter().zip(dst).take(n) {
        let s = apply(ts, *s);
        let d = apply(td, *d);
        let (x, y) = (f64::from(s.x), f64::from(s.y));
        let (u, v) = (f64::from(d.x), f64::from(d.y));
        let rows = [
            ([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y], u),
            ([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y], v),
        ];
        for (a, rhs) in rows {
            for (i, ai) in a.iter().enumerate() {
                for (j, aj) in a.iter().enumerate() {
                    #[allow(clippy::indexing_slicing)] // i, j < 8 by construction
                    {
                        normal[i][j] += ai * aj;
                    }
                }
                #[allow(clippy::indexing_slicing)] // i < 8 by construction
                {
                    normal[i][8] += ai * rhs;
                }
            }
        }
    }

    let h = solve8(normal).ok_or(CalibError::Singular)?;
    #[allow(clippy::cast_possible_truncation)]
    let hn = Mat3::from_cols_array(&[
        h[0] as f32,
        h[3] as f32,
        h[6] as f32, // column 0
        h[1] as f32,
        h[4] as f32,
        h[7] as f32, // column 1
        h[2] as f32,
        h[5] as f32,
        1.0, // column 2
    ]);
    Homography::new(td_inv * hn * ts)
}

#[cfg(test)]
mod tests {
    use fathom_core::PlanePoint;

    use super::*;

    const PLANE: [Vec2; 4] = [
        Vec2::new(0.0, 0.0),
        Vec2::new(1.0, 0.0),
        Vec2::new(1.0, 1.0),
        Vec2::new(0.0, 1.0),
    ];

    #[test]
    fn fits_four_clicked_corners_exactly() {
        let image = [
            Vec2::new(100.0, 100.0),
            Vec2::new(500.0, 120.0),
            Vec2::new(470.0, 400.0),
            Vec2::new(130.0, 380.0),
        ];
        let h = homography_from_correspondences(&PLANE, &image).unwrap();
        for (p, want) in PLANE.iter().zip(&image) {
            let got = warp(PlanePoint::from_repr(*p), &h).0;
            assert!((got - *want).length() < 1e-2, "{got:?} vs {want:?}");
        }
    }

    #[test]
    fn round_trips_through_the_inverse() {
        let image = [
            Vec2::new(100.0, 100.0),
            Vec2::new(500.0, 120.0),
            Vec2::new(470.0, 400.0),
            Vec2::new(130.0, 380.0),
        ];
        let h = homography_from_correspondences(&PLANE, &image).unwrap();
        let p = PlanePoint::new(0.37, 0.62);
        assert!((unwarp(warp(p, &h), &h).0 - p.0).length() < 1e-4);
    }

    #[test]
    fn overdetermined_averages() {
        let image: [Vec2; 5] = [
            Vec2::new(100.0, 100.0),
            Vec2::new(500.0, 120.0),
            Vec2::new(470.0, 400.0),
            Vec2::new(130.0, 380.0),
            Vec2::new(300.0, 240.0),
        ];
        let mut plane = [Vec2::ZERO; 5];
        plane[..4].copy_from_slice(&PLANE);
        plane[4] = Vec2::new(0.5, 0.5);
        assert!(homography_from_correspondences(&plane, &image).is_ok());
    }

    #[test]
    fn degenerate_inputs_are_rejected_not_silently_wrong() {
        assert_eq!(
            homography_from_correspondences(&PLANE[..3], &PLANE[..3]),
            Err(CalibError::TooFewCorrespondences(3))
        );
        let collinear = [
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(3.0, 0.0),
        ];
        assert_eq!(
            homography_from_correspondences(&collinear, &collinear),
            Err(CalibError::Singular)
        );
        let coincident = [Vec2::ZERO; 4];
        assert_eq!(
            homography_from_correspondences(&coincident, &PLANE),
            Err(CalibError::Singular)
        );
    }
}
