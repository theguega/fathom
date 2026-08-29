//! The wgpu backend and drawing primitives for [fathom](https://docs.rs/fathom).
//!
//! Every primitive is a free function over plain slices. There is no scene
//! graph, no retained tree, no registry: state is regenerated every frame, so
//! there is nothing to invalidate and no invalidation bug class. You own the
//! loop, you own the data, and the library draws.
//!
//! Two pipelines carry the whole set. Textured quads and lines: video is a
//! quad, a depth heatmap is a quad, text is quads, a point cloud is quads, a
//! trajectory is lines, an axis triad is lines.
//!
//! ```no_run
//! use fathom_core::{Color, Meters};
//! use fathom_render::{Ctx, Orbit, begin_frame, draw_grid};
//!
//! # fn demo(ctx: &mut Ctx, orbit: &Orbit) {
//! let mut f = begin_frame(ctx);
//! let mut s = f.scene(&orbit.camera(16.0 / 9.0));
//! draw_grid(&mut s, 20, Meters(0.1), Color::GRAY);
//! s.end();
//! f.end();
//! # }
//! ```
#![deny(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

mod batch;
mod camera;
mod ctx;
mod draw2d;
mod draw3d;
mod font;
mod frame;
mod texture;
mod vertex;

pub use camera::{Camera, Orbit};
pub use ctx::{Ctx, InitError};
pub use draw2d::{
    draw_bbox, draw_line_2d, draw_line_strip_2d, draw_polygon, draw_text_at, draw_texture,
    text_width,
};
pub use draw3d::{
    draw_frames, draw_grid, draw_line_3d, draw_line_strip_3d, draw_line_strip_3d_vc,
    draw_points_3d, draw_wire_ellipsoid,
};
pub use frame::{Frame, Scene, begin_frame};
pub use texture::{Filter, Format, Texture, TextureError, update_texture, upload_texture};
pub use vertex::Vertex;
