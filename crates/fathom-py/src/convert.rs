//! Zero-copy views of numpy arrays, and the small value types.
//!
//! Every conversion here either borrows the array's memory outright or fails.
//! Nothing silently copies, and nothing silently reinterprets: a caller who
//! hands over a transposed or Fortran-ordered view finds out at the boundary,
//! not from a plot that is subtly wrong three weeks later.

use fathom_core::{Color, ImagePoint, Rect, WorldPoint};
use numpy::{
    Element, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods as _,
};
use pyo3::{exceptions::PyValueError, prelude::*};

/// Borrow a 2-D array's memory, checking the row width and the memory order.
///
/// `as_slice` alone is not enough: it rejects a strided view but *accepts* a
/// Fortran-ordered one, whose bytes are in the wrong order for a row-major
/// read. That would be a silent transposition, so the layout is checked here
/// explicitly.
fn rows<'a, T: Element>(
    a: &'a PyReadonlyArray2<'a, T>,
    width: usize,
    what: &str,
) -> PyResult<&'a [T]> {
    let shape = a.shape();
    if shape.get(1) != Some(&width) {
        return Err(PyValueError::new_err(format!(
            "expected an (N, {width}) array of {what}, got {shape:?}"
        )));
    }
    contiguous(a.is_c_contiguous(), a.as_slice().ok())
}

/// The one place the contiguity message is written.
fn contiguous<T>(is_c_contiguous: bool, slice: Option<T>) -> PyResult<T> {
    if !is_c_contiguous {
        return Err(PyValueError::new_err(
            "array must be C-contiguous; call np.ascontiguousarray(a) first",
        ));
    }
    slice.ok_or_else(|| {
        PyValueError::new_err("array must be C-contiguous; call np.ascontiguousarray(a) first")
    })
}

/// Borrow an `(N, 3) float32` array as world points, with no copy.
///
/// # Errors
///
/// If the array is not C-contiguous or its second axis is not 3.
pub(crate) fn world_points<'a>(a: &'a PyReadonlyArray2<'a, f32>) -> PyResult<&'a [WorldPoint]> {
    Ok(bytemuck::cast_slice(rows(a, 3, "positions")?))
}

/// Borrow an `(N, 2) float32` array as image points, with no copy.
///
/// # Errors
///
/// If the array is not C-contiguous or its second axis is not 2.
pub(crate) fn image_points<'a>(a: &'a PyReadonlyArray2<'a, f32>) -> PyResult<&'a [ImagePoint]> {
    Ok(bytemuck::cast_slice(rows(a, 2, "pixels")?))
}

/// Borrow an `(N, 4) uint8` array as colors, with no copy.
///
/// # Errors
///
/// If the array is not C-contiguous or its second axis is not 4.
pub(crate) fn colors<'a>(a: &'a PyReadonlyArray2<'a, u8>) -> PyResult<&'a [Color]> {
    Ok(bytemuck::cast_slice(rows(a, 4, "RGBA")?))
}

/// Borrow a square `(n, n) float32` matrix, with no copy.
///
/// # Errors
///
/// If the array is not C-contiguous or not `n` by `n`.
pub(crate) fn square<'a>(a: &'a PyReadonlyArray2<'a, f32>, n: usize) -> PyResult<&'a [f32]> {
    let shape = a.shape();
    if shape != [n, n] {
        return Err(PyValueError::new_err(format!(
            "expected a ({n}, {n}) matrix, got {shape:?}"
        )));
    }
    contiguous(a.is_c_contiguous(), a.as_slice().ok())
}

/// Borrow an `(N, 4, 4) float32` array of transforms, with no copy.
///
/// # Errors
///
/// If the array is not C-contiguous or is not shaped `(N, 4, 4)`.
pub(crate) fn transforms<'a>(a: &'a PyReadonlyArray3<'a, f32>) -> PyResult<&'a [f32]> {
    let shape = a.shape();
    if shape.get(1) != Some(&4) || shape.get(2) != Some(&4) {
        return Err(PyValueError::new_err(format!(
            "expected an (N, 4, 4) array of transforms, got {shape:?}"
        )));
    }
    contiguous(a.is_c_contiguous(), a.as_slice().ok())
}

/// Borrow a 1-D float32 array, with no copy.
///
/// # Errors
///
/// If the array is not C-contiguous.
pub(crate) fn scalars<'a>(a: &'a PyReadonlyArray1<'a, f32>) -> PyResult<&'a [f32]> {
    contiguous(a.is_c_contiguous(), a.as_slice().ok())
}

/// `(r, g, b, a)` as Python sees colors, 0-255.
pub(crate) fn color(rgba: (u8, u8, u8, u8)) -> Color {
    Color::rgba(rgba.0, rgba.1, rgba.2, rgba.3)
}

/// `(x, y, w, h)` in pixels.
pub(crate) fn rect(r: (f32, f32, f32, f32)) -> Rect {
    Rect::new(r.0, r.1, r.2, r.3)
}
