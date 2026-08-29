//! Screen and image space primitives.
//!
//! Every one of these lowers to textured quads or to line segments. Adding a
//! primitive here means writing a lowering function, never a new shader path.

use fathom_core::{Color, ImagePoint, Rect, TextScale};
use glam::{Vec2, Vec4};

use crate::{
    Frame, Texture, font,
    vertex::{Topology, Vertex},
};

/// Emit one textured quad, corners given clockwise from the top left.
fn quad(f: &mut Frame<'_>, corners: [Vec4; 4], uvs: [Vec2; 4], color: Color) {
    for i in [0, 1, 2, 0, 2, 3] {
        let (Some(&clip), Some(&uv)) = (corners.get(i), uvs.get(i)) else {
            continue;
        };
        f.push(Vertex::new(clip, uv, color));
    }
}

/// The four corners of a rectangle, in clip space, clockwise from top left.
fn rect_corners(f: &Frame<'_>, r: Rect) -> [Vec4; 4] {
    [
        f.clip(Vec2::new(r.x, r.y)),
        f.clip(Vec2::new(r.right(), r.y)),
        f.clip(Vec2::new(r.right(), r.bottom())),
        f.clip(Vec2::new(r.x, r.bottom())),
    ]
}

const FULL_UV: [Vec2; 4] = [
    Vec2::new(0.0, 0.0),
    Vec2::new(1.0, 0.0),
    Vec2::new(1.0, 1.0),
    Vec2::new(0.0, 1.0),
];

/// Draw a texture into a rectangle, multiplied by `tint`.
///
/// This is the video primitive, the depth-heatmap primitive and the
/// similarity-matrix primitive, all at once: a heatmap is a texture, so there
/// is no heatmap primitive.
///
/// Use [`Rect::fit_aspect`] with [`Texture::aspect`] to letterbox rather than
/// stretch:
///
/// ```no_run
/// # use fathom_render::{Frame, Texture, draw_texture};
/// # use fathom_core::Color;
/// # fn demo(f: &mut Frame<'_>, camera: &Texture) {
/// let dst = f.viewport().fit_aspect(camera.aspect());
/// draw_texture(f, camera, dst, Color::WHITE);
/// # }
/// ```
pub fn draw_texture(f: &mut Frame<'_>, tex: &Texture, dst: Rect, tint: Color) {
    let corners = rect_corners(f, dst);
    f.begin(Topology::Triangles, Some(&tex.bind_group), 6);
    quad(f, corners, FULL_UV, tint);
}

/// Draw one line segment in pixels.
pub fn draw_line_2d(f: &mut Frame<'_>, a: ImagePoint, b: ImagePoint, color: Color) {
    let uv = font::white_uv();
    let (a, b) = (f.clip(a.0), f.clip(b.0));
    f.begin(Topology::Lines, None, 2);
    f.push(Vertex::new(a, uv, color));
    f.push(Vertex::new(b, uv, color));
}

/// Draw a connected polyline in pixels.
///
/// This is the whole plotting subsystem: a time series is one of these per
/// channel, with the caller's own axis scaling. Fewer than two points draws
/// nothing.
///
/// ```no_run
/// # use fathom_render::{Frame, draw_line_strip_2d};
/// # use fathom_core::{Color, ImagePoint, Rect};
/// # fn demo(f: &mut Frame<'_>, samples: &[f32], plot: Rect, scratch: &mut Vec<ImagePoint>) {
/// scratch.clear();
/// scratch.extend(samples.iter().enumerate().map(|(i, &v)| {
///     # #[allow(clippy::cast_precision_loss)]
///     let x = plot.x + plot.w * i as f32 / samples.len() as f32;
///     ImagePoint::new(x, plot.bottom() - v * plot.h)
/// }));
/// draw_line_strip_2d(f, scratch, Color::GREEN);
/// # }
/// ```
pub fn draw_line_strip_2d(f: &mut Frame<'_>, pts: &[ImagePoint], color: Color) {
    let uv = font::white_uv();
    let per_batch = (f.batch_limit() / 2).max(1);
    let segments = pts.len().saturating_sub(1);
    let mut done = 0;

    while done < segments {
        let n = (segments - done).min(per_batch);
        f.begin(Topology::Lines, None, n * 2);
        for i in done..done + n {
            let (Some(a), Some(b)) = (pts.get(i), pts.get(i + 1)) else {
                continue;
            };
            let (a, b) = (f.clip(a.0), f.clip(b.0));
            f.push(Vertex::new(a, uv, color));
            f.push(Vertex::new(b, uv, color));
        }
        done += n;
    }
}

/// Draw a rectangle outline: the detection-box primitive.
pub fn draw_bbox(f: &mut Frame<'_>, r: Rect, color: Color) {
    let uv = font::white_uv();
    let c = rect_corners(f, r);
    f.begin(Topology::Lines, None, 8);
    for i in 0..4 {
        let (Some(&a), Some(&b)) = (c.get(i), c.get((i + 1) % 4)) else {
            continue;
        };
        f.push(Vertex::new(a, uv, color));
        f.push(Vertex::new(b, uv, color));
    }
}

/// Fill a convex polygon in pixels, as a triangle fan from the first vertex.
///
/// Concave outlines render wrong; that is the trade for having no tessellator.
/// Fewer than three points draws nothing.
pub fn draw_polygon(f: &mut Frame<'_>, pts: &[ImagePoint], color: Color) {
    let uv = font::white_uv();
    let Some(first) = pts.first() else { return };
    let anchor = f.clip(first.0);
    let per_batch = (f.batch_limit() / 3).max(1);
    let triangles = pts.len().saturating_sub(2);
    let mut done = 0;

    while done < triangles {
        let n = (triangles - done).min(per_batch);
        f.begin(Topology::Triangles, None, n * 3);
        for i in done..done + n {
            let (Some(b), Some(c)) = (pts.get(i + 1), pts.get(i + 2)) else {
                continue;
            };
            let (b, c) = (f.clip(b.0), f.clip(c.0));
            f.push(Vertex::new(anchor, uv, color));
            f.push(Vertex::new(b, uv, color));
            f.push(Vertex::new(c, uv, color));
        }
        done += n;
    }
}

/// Width in pixels that [`draw_text_at`] will occupy, for laying out a HUD.
#[must_use]
#[allow(clippy::cast_precision_loss)] // a string long enough to lose precision would not fit on a screen
pub fn text_width(text: &str, scale: TextScale) -> f32 {
    let cells = text.chars().count() as f32;
    cells * font::GLYPH as f32 * scale.factor()
}

/// Draw a line of text, one quad per glyph, with `pos` at its top-left corner.
///
/// ASCII and Latin-1; anything else renders as `?`. Integer scaling keeps the
/// bitmap crisp. Newlines are not handled - a line is a line, and stacking them
/// is a `y += 8.0 * scale.factor()` in your loop.
///
/// ```no_run
/// # use fathom_render::{Frame, draw_text_at};
/// # use fathom_core::{Color, ImagePoint, TextScale};
/// # fn demo(f: &mut Frame<'_>, fps: f32, t: f64) {
/// draw_text_at(f, ImagePoint::new(8.0, 8.0), &format!("{fps:5.1} fps"), TextScale::X2, Color::WHITE);
/// draw_text_at(f, ImagePoint::new(8.0, 24.0), &format!("t = {t:.3}s"), TextScale::X2, Color::GRAY);
/// # }
/// ```
pub fn draw_text_at(
    f: &mut Frame<'_>,
    pos: ImagePoint,
    text: &str,
    scale: TextScale,
    color: Color,
) {
    #[allow(clippy::cast_precision_loss)]
    let cell = font::GLYPH as f32 * scale.factor();
    #[allow(clippy::cast_precision_loss)]
    let atlas = font::ATLAS as f32;
    #[allow(clippy::cast_precision_loss)]
    let g = font::GLYPH as f32;

    let per_batch = (f.batch_limit() / 6).max(1);
    let mut cursor = pos.0;
    let total = text.chars().count();
    let mut chars = text.chars();
    let mut done = 0;

    while done < total {
        let n = (total - done).min(per_batch);
        f.begin(Topology::Triangles, None, n * 6);
        for _ in 0..n {
            let Some(c) = chars.next() else { break };
            #[allow(clippy::cast_precision_loss)]
            let (ox, oy) = {
                let (x, y) = font::glyph_origin(font::glyph_index(c));
                (x as f32, y as f32)
            };
            let (u0, v0) = (ox / atlas, oy / atlas);
            let (u1, v1) = ((ox + g) / atlas, (oy + g) / atlas);
            let r = Rect::new(cursor.x, cursor.y, cell, cell);
            let corners = rect_corners(f, r);
            quad(
                f,
                corners,
                [
                    Vec2::new(u0, v0),
                    Vec2::new(u1, v0),
                    Vec2::new(u1, v1),
                    Vec2::new(u0, v1),
                ],
                color,
            );
            cursor.x += cell;
        }
        done += n;
    }
}
