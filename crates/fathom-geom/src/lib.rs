//! Pure geometry and color math for [fathom](https://docs.rs/fathom).
//!
//! No GPU, no allocation on the hot paths, no `std`. Everything here is a
//! **verb over slices** or a closed-form transform, which is why it is testable
//! and benchmarkable without opening a window.
//!
//! ```
//! use fathom_geom::{ColorMap, Extrinsics, Intrinsics, WorldPoint, colormap, project};
//!
//! let k = Intrinsics::new(600.0, 600.0, 320.0, 240.0)?;
//! let px = project(WorldPoint::new(0.0, 0.0, 1.0), &k, &Extrinsics::IDENTITY);
//! assert_eq!(px.map(|p| p.0.to_array()), Some([320.0, 240.0]));
//!
//! let colors = colormap(&[0.0, 0.5, 1.0], 0.0..1.0, ColorMap::Viridis);
//! assert_eq!(colors.len(), 3);
//! # Ok::<_, fathom_core::CalibError>(())
//! ```
#![no_std]
#![deny(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

extern crate alloc;

mod color;
mod homography;
mod pinhole;

pub use color::{colormap, colormap_into};
pub use fathom_core::*;
pub use homography::{homography_from_correspondences, unwarp, warp};
pub use pinhole::{project, unproject};
