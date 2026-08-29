//! The pure-maths functions: colormaps and camera geometry.
//!
//! No GPU, no renderer needed. These work on their own, which is the point of
//! `fathom-geom` being a separate crate.

use fathom_core::{Extrinsics, ImagePoint, Intrinsics, Mat3, Mat4, Meters, PlanePoint};
use fathom_geom as fg;
use numpy::{PyArray2, PyArrayMethods as _, PyReadonlyArray1, PyReadonlyArray2, ToPyArray as _};
use pyo3::{exceptions::PyValueError, prelude::*};

/// Register every maths function on the module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(colormap, m)?)?;
    m.add_function(wrap_pyfunction!(project, m)?)?;
    m.add_function(wrap_pyfunction!(unproject, m)?)?;
    m.add_function(wrap_pyfunction!(warp, m)?)?;
    m.add_function(wrap_pyfunction!(unwarp, m)?)?;
    m.add_function(wrap_pyfunction!(homography_from_correspondences, m)?)?;
    Ok(())
}

fn ramp(name: &str) -> PyResult<fathom_core::ColorMap> {
    match name.to_ascii_lowercase().as_str() {
        "turbo" => Ok(fathom_core::ColorMap::Turbo),
        "viridis" => Ok(fathom_core::ColorMap::Viridis),
        "magma" => Ok(fathom_core::ColorMap::Magma),
        "grey" | "gray" => Ok(fathom_core::ColorMap::Grey),
        "coolwarm" => Ok(fathom_core::ColorMap::Coolwarm),
        other => Err(PyValueError::new_err(format!(
            "unknown colormap {other:?}; expected turbo, viridis, magma, grey or coolwarm"
        ))),
    }
}

fn intrinsics(k: (f32, f32, f32, f32), dist: Option<[f32; 5]>) -> PyResult<Intrinsics> {
    let base =
        Intrinsics::new(k.0, k.1, k.2, k.3).map_err(|e| PyValueError::new_err(e.to_string()))?;
    match dist {
        None => Ok(base),
        Some([k1, k2, k3, p1, p2]) => base
            .with_brown_conrady([k1, k2, k3], [p1, p2])
            .map_err(|e| PyValueError::new_err(e.to_string())),
    }
}

fn extrinsics(m: &PyReadonlyArray2<'_, f32>) -> PyResult<Extrinsics> {
    let slice = crate::convert::square(m, 4)?;
    Extrinsics::from_world_to_camera(Mat4::from_cols_slice(slice))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Map scalars through a perceptual ramp, returning `(N, 4)` uint8 RGBA.
///
/// One function for depth maps, attention heatmaps, similarity matrices and
/// outlier scores. Values outside `(vmin, vmax)` clamp; NaN maps to the low end.
///
/// Args:
///     values: 1-D float32 array.
///     vmin: low end of the range.
///     vmax: high end of the range.
///     cmap: "turbo", "viridis", "magma", "grey" or "coolwarm".
#[pyfunction]
#[pyo3(signature = (values, vmin=0.0, vmax=1.0, cmap="turbo"))]
fn colormap<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f32>,
    vmin: f32,
    vmax: f32,
    cmap: &str,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let map = ramp(cmap)?;
    let values = crate::convert::scalars(&values)?;
    let colors = fg::colormap(values, vmin..vmax, map);
    let bytes: &[u8] = bytemuck::cast_slice(&colors);
    let arr = bytes.to_pyarray(py);
    arr.reshape([colors.len(), 4])
}

/// Project world points into a calibrated camera image.
///
/// Returns `(N, 2)` float32 pixels; points at or behind the image plane come
/// back as NaN, which is a real geometric outcome rather than an error.
///
/// Args:
///     points: `(N, 3)` float32 array of world positions.
///     k: `(fx, fy, cx, cy)`.
///     world_to_camera: `(4, 4)` float32 matrix, column-major.
///     distortion: optional `(k1, k2, k3, p1, p2)` Brown-Conrady coefficients.
#[pyfunction]
#[pyo3(signature = (points, k, world_to_camera, distortion=None))]
fn project<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f32>,
    k: (f32, f32, f32, f32),
    world_to_camera: PyReadonlyArray2<'py, f32>,
    distortion: Option<[f32; 5]>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let k = intrinsics(k, distortion)?;
    let e = extrinsics(&world_to_camera)?;
    let points = crate::convert::world_points(&points)?;

    let mut out = Vec::with_capacity(points.len() * 2);
    for p in points {
        let px = fg::project(*p, &k, &e).map_or([f32::NAN, f32::NAN], |q| q.0.to_array());
        out.extend_from_slice(&px);
    }
    let n = points.len();
    out.to_pyarray(py).reshape([n, 2])
}

/// Lift pixels back into the world, given the depth along the optical axis.
///
/// Args:
///     pixels: `(N, 2)` float32 array.
///     depth: 1-D float32 array of metres, one per pixel.
#[pyfunction]
#[pyo3(signature = (pixels, depth, k, world_to_camera, distortion=None))]
fn unproject<'py>(
    py: Python<'py>,
    pixels: PyReadonlyArray2<'py, f32>,
    depth: PyReadonlyArray1<'py, f32>,
    k: (f32, f32, f32, f32),
    world_to_camera: PyReadonlyArray2<'py, f32>,
    distortion: Option<[f32; 5]>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let k = intrinsics(k, distortion)?;
    let e = extrinsics(&world_to_camera)?;
    let pixels = crate::convert::image_points(&pixels)?;
    let depth = crate::convert::scalars(&depth)?;
    if pixels.len() != depth.len() {
        return Err(PyValueError::new_err(format!(
            "pixels and depth must be the same length, got {} and {}",
            pixels.len(),
            depth.len()
        )));
    }

    let mut out = Vec::with_capacity(pixels.len() * 3);
    for (px, z) in pixels.iter().zip(depth) {
        out.extend_from_slice(&fg::unproject(*px, Meters(*z), &k, &e).0.to_array());
    }
    let n = pixels.len();
    out.to_pyarray(py).reshape([n, 3])
}

fn homography(m: &PyReadonlyArray2<'_, f32>) -> PyResult<fathom_core::Homography> {
    fathom_core::Homography::new(Mat3::from_cols_slice(crate::convert::square(m, 3)?))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Map points on the physical plane to their pixels in the image.
#[pyfunction]
#[pyo3(signature = (points, h))]
fn warp<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f32>,
    h: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let h = homography(&h)?;
    let points = crate::convert::image_points(&points)?;
    let mut out = Vec::with_capacity(points.len() * 2);
    for p in points {
        let q = fg::warp(PlanePoint::from_repr(p.0), &h);
        out.extend_from_slice(&q.0.to_array());
    }
    let n = points.len();
    out.to_pyarray(py).reshape([n, 2])
}

/// Map pixels back onto the physical plane.
#[pyfunction]
#[pyo3(signature = (pixels, h))]
fn unwarp<'py>(
    py: Python<'py>,
    pixels: PyReadonlyArray2<'py, f32>,
    h: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let h = homography(&h)?;
    let pixels = crate::convert::image_points(&pixels)?;
    let mut out = Vec::with_capacity(pixels.len() * 2);
    for p in pixels {
        out.extend_from_slice(&fg::unwarp(ImagePoint::from_repr(p.0), &h).0.to_array());
    }
    let n = pixels.len();
    out.to_pyarray(py).reshape([n, 2])
}

/// Fit a homography to four or more plane-to-image correspondences.
///
/// Returns the `(3, 3)` float32 matrix, column-major. Degenerate input -
/// collinear or coincident points - raises rather than returning a silently
/// wrong matrix.
#[pyfunction]
#[pyo3(signature = (src, dst))]
fn homography_from_correspondences<'py>(
    py: Python<'py>,
    src: PyReadonlyArray2<'py, f32>,
    dst: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let src = crate::convert::image_points(&src)?;
    let dst = crate::convert::image_points(&dst)?;
    let src: Vec<fathom_core::Vec2> = src.iter().map(|p| p.0).collect();
    let dst: Vec<fathom_core::Vec2> = dst.iter().map(|p| p.0).collect();

    let h = fg::homography_from_correspondences(&src, &dst)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    h.forward().to_cols_array().to_pyarray(py).reshape([3, 3])
}
