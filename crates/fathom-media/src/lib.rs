//! The export sink for [fathom](https://docs.rs/fathom): rendered frames in,
//! an encoded video file out.
//!
//! Encode only. Decode is not here and never will be: it belongs to the adapter
//! crate that already knows the container layout, while "render N frames to an
//! mp4" is an output of the renderer and so belongs to the renderer's side of
//! the line.
//!
//! # Why a process, not a C library
//!
//! This crate spawns the `ffmpeg` binary and writes raw frames to its stdin.
//! It links no C libraries, which means it has **no dependencies at all** - the
//! default build stays pure Rust, docs.rs builds clean, and Python wheels do
//! not need a manylinux ffmpeg. It also works against whatever ffmpeg the user
//! already has, rather than the narrow version range an FFI binding pins.
//!
//! The trade is that `ffmpeg` must be on `PATH` at runtime, which
//! [`Encoder::new`] checks and reports immediately rather than at the first
//! frame.
//!
//! ```no_run
//! use fathom_media::{Encoder, EncodeOptions};
//!
//! # fn demo(frames: &[Vec<u8>]) -> Result<(), fathom_media::MediaError> {
//! let mut enc = Encoder::new("out.mp4", 1280, 720, &EncodeOptions::default())?;
//! for rgba in frames {
//!     enc.write(rgba)?;
//! }
//! let done = enc.finish()?;
//! println!("wrote {} frames", done.frames());
//! # Ok(()) }
//! ```
#![deny(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

mod encoder;

pub use encoder::{EncodeOptions, Encoder, Finished, MediaError, Open};
