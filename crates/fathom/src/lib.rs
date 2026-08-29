#![doc = include_str!("../README.md")]
#![deny(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

// `Camera` means two different things in the two crates below: a coordinate
// frame in `fathom-core`, a viewpoint in `fathom-render`. Rather than let a
// glob pick a winner, the frame keeps the longer name here - it is almost
// always spelled `CameraPoint` anyway - and `Camera` means the thing you bind
// to a `Scene`.
pub use fathom_core::{
    CalibError, Camera as CameraFrame, CameraPoint, Color, ColorMap, CoordFrame, Extrinsics,
    FrameIdx, Homography, Image, ImagePoint, Intrinsics, Mat3, Mat4, Meters, Plane, PlanePoint,
    Point, Quat, Radians, Rect, TextScale, Timestamp, Vec2, Vec3, World, WorldPoint, look_at,
    point,
};
pub use fathom_geom::{
    colormap, colormap_into, homography_from_correspondences, project, unproject, unwarp, warp,
};
pub use fathom_render::*;

/// mp4 export: rendered frames in, an encoded file out.
///
/// Off by default. It spawns the `ffmpeg` binary rather than linking a C
/// library, so enabling it adds no dependencies and no build-time toolchain -
/// only a runtime requirement that `ffmpeg` be on `PATH`.
#[cfg(feature = "media")]
#[cfg_attr(docsrs, doc(cfg(feature = "media")))]
pub mod media {
    pub use fathom_media::*;
}

/// Everything fathom exposes, in one `use`.
///
/// The whole public surface is about forty names and every one of them is
/// meant to be reached for, so the prelude is not a curated subset: it is the
/// crate.
///
/// ```
/// use fathom::prelude::*;
///
/// let path = [WorldPoint::new(0.0, 0.0, 0.0), WorldPoint::new(0.1, 0.0, 0.0)];
/// let colors = colormap(&[0.0, 1.0], 0.0..1.0, ColorMap::Turbo);
/// assert_eq!(colors.len(), path.len());
/// ```
pub mod prelude {
    pub use crate::*;
}
