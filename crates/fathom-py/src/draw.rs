//! The drawing primitives, mirrored one for one.

use fathom_core::{ImagePoint, Meters, TextScale, WorldPoint};
use fathom_render as fr;
use numpy::{PyReadonlyArray2, PyReadonlyArray3};
use pyo3::{exceptions::PyValueError, prelude::*};

use crate::{
    convert,
    renderer::{Renderer, Texture},
};

/// Register every drawing function on the module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(draw_texture, m)?)?;
    m.add_function(wrap_pyfunction!(draw_line_2d, m)?)?;
    m.add_function(wrap_pyfunction!(draw_line_strip_2d, m)?)?;
    m.add_function(wrap_pyfunction!(draw_bbox, m)?)?;
    m.add_function(wrap_pyfunction!(draw_polygon, m)?)?;
    m.add_function(wrap_pyfunction!(draw_text_at, m)?)?;
    m.add_function(wrap_pyfunction!(text_width, m)?)?;
    m.add_function(wrap_pyfunction!(draw_line_3d, m)?)?;
    m.add_function(wrap_pyfunction!(draw_line_strip_3d, m)?)?;
    m.add_function(wrap_pyfunction!(draw_line_strip_3d_vc, m)?)?;
    m.add_function(wrap_pyfunction!(draw_points_3d, m)?)?;
    m.add_function(wrap_pyfunction!(draw_grid, m)?)?;
    m.add_function(wrap_pyfunction!(draw_frames, m)?)?;
    m.add_function(wrap_pyfunction!(draw_wire_ellipsoid, m)?)?;
    Ok(())
}

fn scale(px: u32) -> TextScale {
    match px {
        0 | 1 => TextScale::X1,
        2 => TextScale::X2,
        3 => TextScale::X3,
        _ => TextScale::X4,
    }
}

/// Draw a texture into a rectangle, multiplied by `tint`.
///
/// A heatmap is a texture, so there is no heatmap primitive: run your scalars
/// through `colormap` and upload the result.
///
/// Args:
///     r: the renderer, with a frame open.
///     tex: the texture to draw.
///     dst: `(x, y, w, h)` in pixels.
///     tint: `(r, g, b, a)`, 0-255.
#[pyfunction]
#[pyo3(signature = (r, tex, dst, tint=(255, 255, 255, 255)))]
fn draw_texture(
    r: &mut Renderer,
    tex: &Texture,
    dst: (f32, f32, f32, f32),
    tint: (u8, u8, u8, u8),
) -> PyResult<()> {
    let (dst, tint) = (convert::rect(dst), convert::color(tint));
    fr::draw_texture(r.frame()?, &tex.0, dst, tint);
    Ok(())
}

/// Draw one line segment in pixels.
#[pyfunction]
#[pyo3(signature = (r, a, b, color))]
fn draw_line_2d(
    r: &mut Renderer,
    a: (f32, f32),
    b: (f32, f32),
    color: (u8, u8, u8, u8),
) -> PyResult<()> {
    let color = convert::color(color);
    fr::draw_line_2d(
        r.frame()?,
        ImagePoint::new(a.0, a.1),
        ImagePoint::new(b.0, b.1),
        color,
    );
    Ok(())
}

/// Draw a connected polyline in pixels.
///
/// This is the whole plotting subsystem: a time series is one call per channel,
/// with your own axis scaling.
///
/// Args:
///     pts: `(N, 2)` float32 array of pixel positions.
#[pyfunction]
#[pyo3(signature = (r, pts, color))]
fn draw_line_strip_2d(
    r: &mut Renderer,
    pts: PyReadonlyArray2<'_, f32>,
    color: (u8, u8, u8, u8),
) -> PyResult<()> {
    let color = convert::color(color);
    let pts = convert::image_points(&pts)?;
    fr::draw_line_strip_2d(r.frame()?, pts, color);
    Ok(())
}

/// Draw a rectangle outline: the detection-box primitive.
#[pyfunction]
#[pyo3(signature = (r, rect, color))]
fn draw_bbox(
    r: &mut Renderer,
    rect: (f32, f32, f32, f32),
    color: (u8, u8, u8, u8),
) -> PyResult<()> {
    let (rect, color) = (convert::rect(rect), convert::color(color));
    fr::draw_bbox(r.frame()?, rect, color);
    Ok(())
}

/// Fill a convex polygon in pixels.
///
/// Args:
///     pts: `(N, 2)` float32 array of pixel positions.
#[pyfunction]
#[pyo3(signature = (r, pts, color))]
fn draw_polygon(
    r: &mut Renderer,
    pts: PyReadonlyArray2<'_, f32>,
    color: (u8, u8, u8, u8),
) -> PyResult<()> {
    let color = convert::color(color);
    let pts = convert::image_points(&pts)?;
    fr::draw_polygon(r.frame()?, pts, color);
    Ok(())
}

/// Draw a line of text, with `pos` at its top-left corner.
///
/// ASCII and Latin-1; anything else renders as `?`. `size` is an integer
/// magnification of the 8px cell, 1 to 4.
#[pyfunction]
#[pyo3(signature = (r, pos, text, size=1, color=(255, 255, 255, 255)))]
fn draw_text_at(
    r: &mut Renderer,
    pos: (f32, f32),
    text: &str,
    size: u32,
    color: (u8, u8, u8, u8),
) -> PyResult<()> {
    let color = convert::color(color);
    fr::draw_text_at(
        r.frame()?,
        ImagePoint::new(pos.0, pos.1),
        text,
        scale(size),
        color,
    );
    Ok(())
}

/// Width in pixels that `draw_text_at` will occupy, for laying out a HUD.
#[pyfunction]
#[pyo3(signature = (text, size=1))]
fn text_width(text: &str, size: u32) -> f32 {
    fr::text_width(text, scale(size))
}

/// Draw one line segment between two world points.
#[pyfunction]
#[pyo3(signature = (r, a, b, color))]
fn draw_line_3d(
    r: &mut Renderer,
    a: (f32, f32, f32),
    b: (f32, f32, f32),
    color: (u8, u8, u8, u8),
) -> PyResult<()> {
    let color = convert::color(color);
    fr::draw_line_3d(
        &mut r.bound_scene()?,
        WorldPoint::new(a.0, a.1, a.2),
        WorldPoint::new(b.0, b.1, b.2),
        color,
    );
    Ok(())
}

/// Draw a connected polyline through world points.
///
/// Args:
///     pts: `(N, 3)` float32 array of positions, borrowed zero-copy.
#[pyfunction]
#[pyo3(signature = (r, pts, color))]
fn draw_line_strip_3d(
    r: &mut Renderer,
    pts: PyReadonlyArray2<'_, f32>,
    color: (u8, u8, u8, u8),
) -> PyResult<()> {
    let color = convert::color(color);
    let pts = convert::world_points(&pts)?;
    fr::draw_line_strip_3d(&mut r.bound_scene()?, pts, color);
    Ok(())
}

/// Draw a polyline with a colour per vertex.
///
/// The highest-leverage primitive in the set: outlier scoring, uncertainty
/// bands, time gradients and per-step error are all this one call with a
/// different colour array.
///
/// Args:
///     pts: `(N, 3)` float32 array of positions.
///     colors: `(N, 4)` uint8 array of RGBA.
#[pyfunction]
#[pyo3(signature = (r, pts, colors))]
fn draw_line_strip_3d_vc(
    r: &mut Renderer,
    pts: PyReadonlyArray2<'_, f32>,
    colors: numpy::PyReadonlyArray2<'_, u8>,
) -> PyResult<()> {
    let verts = zip_verts(&pts, &colors)?;
    fr::draw_line_strip_3d_vc(&mut r.bound_scene()?, &verts);
    Ok(())
}

/// Draw a point cloud as camera-facing squares of a metric size.
///
/// Args:
///     pts: `(N, 3)` float32 array of positions.
///     colors: `(N, 4)` uint8 array of RGBA.
///     size: edge length in metres.
#[pyfunction]
#[pyo3(signature = (r, pts, colors, size=0.01))]
fn draw_points_3d(
    r: &mut Renderer,
    pts: PyReadonlyArray2<'_, f32>,
    colors: numpy::PyReadonlyArray2<'_, u8>,
    size: f32,
) -> PyResult<()> {
    let verts = zip_verts(&pts, &colors)?;
    fr::draw_points_3d(&mut r.bound_scene()?, &verts, Meters(size));
    Ok(())
}

/// Draw a ground grid of `slices` cells per side, centred on the origin.
#[pyfunction]
#[pyo3(signature = (r, slices=20, spacing=0.1, color=(128, 128, 128, 255)))]
fn draw_grid(r: &mut Renderer, slices: u32, spacing: f32, color: (u8, u8, u8, u8)) -> PyResult<()> {
    let color = convert::color(color);
    fr::draw_grid(&mut r.bound_scene()?, slices, Meters(spacing), color);
    Ok(())
}

/// Draw an RGB axis triad per transform, connected in order.
///
/// Takes the link transforms your control stack already computed. There is no
/// forward kinematics in fathom, on purpose.
///
/// Args:
///     transforms: `(N, 4, 4)` float32 array, column-major, borrowed zero-copy.
#[pyfunction]
#[pyo3(signature = (r, transforms, axis_len=0.05))]
fn draw_frames(
    r: &mut Renderer,
    transforms: PyReadonlyArray3<'_, f32>,
    axis_len: f32,
) -> PyResult<()> {
    let mats: &[fathom_core::Mat4] = bytemuck::cast_slice(convert::transforms(&transforms)?);
    fr::draw_frames(&mut r.bound_scene()?, mats, Meters(axis_len));
    Ok(())
}

/// Draw a wireframe ellipsoid: covariance, variance, an uncertainty volume.
///
/// Args:
///     axes: `(3, 3)` float32 array mapping the unit sphere to the shell, which
///         for a 1-sigma volume is the Cholesky factor of the covariance.
#[pyfunction]
#[pyo3(signature = (r, center, axes, color))]
fn draw_wire_ellipsoid(
    r: &mut Renderer,
    center: (f32, f32, f32),
    axes: PyReadonlyArray2<'_, f32>,
    color: (u8, u8, u8, u8),
) -> PyResult<()> {
    let m = fathom_core::Mat3::from_cols_slice(convert::square(&axes, 3)?);
    let color = convert::color(color);
    fr::draw_wire_ellipsoid(
        &mut r.bound_scene()?,
        WorldPoint::new(center.0, center.1, center.2),
        m,
        color,
    );
    Ok(())
}

/// Zip positions and colours into the one slice the draw calls take.
///
/// The Rust API takes `&[(WorldPoint, Color)]` precisely so a length mismatch
/// cannot be represented. numpy hands over two arrays, so the check happens
/// here, once, at the boundary.
fn zip_verts(
    pts: &PyReadonlyArray2<'_, f32>,
    colors: &numpy::PyReadonlyArray2<'_, u8>,
) -> PyResult<Vec<(WorldPoint, fathom_core::Color)>> {
    let pts = convert::world_points(pts)?;
    let colors = convert::colors(colors)?;
    if pts.len() != colors.len() {
        return Err(PyValueError::new_err(format!(
            "positions and colors must be the same length, got {} and {}",
            pts.len(),
            colors.len()
        )));
    }
    Ok(pts.iter().copied().zip(colors.iter().copied()).collect())
}
