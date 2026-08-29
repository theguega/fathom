//! Python bindings for [fathom](https://docs.rs/fathom).
//!
//! The Rust free functions, mirrored verbatim. No classes beyond the handles
//! that must exist, no "Pythonic" wrapper layer: that layer is exactly where
//! bloat re-enters, and the whole point of the Rust API is that it is already
//! small enough to use directly.
//!
//! Contiguous numpy arrays cross the boundary zero-copy through
//! `bytemuck::cast_slice`. Non-contiguous input errors out loudly rather than
//! silently copying.
#![deny(missing_docs)]
// This is the FFI surface, and the one crate where `unsafe` is expected. It is
// confined to `renderer.rs`, where a single documented block extends the
// lifetime of a `Frame` that Python holds on the Rust side.
#![allow(unsafe_code)]
// `#[pyfunction]` extracts its arguments by value, so a borrowed numpy array
// arrives owned whether or not the body consumes it. The borrow of the
// underlying buffer is what matters, and that is still zero-copy.
#![allow(clippy::needless_pass_by_value)]
// The doc comments in this crate become Python docstrings verbatim, so they use
// Python's names and Python's punctuation rather than rustdoc markup.
#![allow(clippy::doc_markdown)]

mod convert;
mod draw;
mod math;
mod renderer;

use pyo3::prelude::*;

/// The `fathom` extension module.
#[pymodule]
fn fathom(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<renderer::Renderer>()?;
    m.add_class::<renderer::Texture>()?;
    m.add_class::<renderer::Orbit>()?;
    m.add_class::<renderer::Camera>()?;

    draw::register(m)?;
    math::register(m)?;

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
