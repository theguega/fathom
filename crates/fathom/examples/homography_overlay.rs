//! The uncalibrated path, end to end.
//!
//! Four clicked correspondences give a homography; a workcell floor grid is
//! then warped onto a fixed overhead camera view. No intrinsics, no
//! calibration rig, no mode flag: a caller with a calibrated setup calls
//! `project` instead, and neither pays for the other.
//!
//! Click the four corners of the marked quad, in order: top-left, top-right,
//! bottom-right, bottom-left.
//!
//! Run with `cargo run -p fathom --example homography_overlay`.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
//  ^ example code: every cast here is small and deliberate

mod common;

use std::{error::Error, num::NonZeroU32};

use fathom::prelude::*;

const W: u32 = 320;
const H: u32 = 240;

/// A fake overhead camera frame with a visible ground quad to click on.
#[allow(clippy::indexing_slicing)] // four corners, indexed modulo four
fn overhead_frame() -> Vec<u8> {
    let corners = [(70.0, 60.0), (250.0, 80.0), (230.0, 200.0), (60.0, 175.0)];
    let mut px = vec![0u8; (W * H * 4) as usize];
    for (i, texel) in px.chunks_exact_mut(4).enumerate() {
        let (x, y) = ((i as u32 % W) as f32, (i as u32 / W) as f32);
        // Point-in-quad by winding, so the floor plate is visible.
        let inside = (0..4).all(|k| {
            let (ax, ay) = corners[k];
            let (bx, by) = corners[(k + 1) % 4];
            (bx - ax) * (y - ay) - (by - ay) * (x - ax) > 0.0
        });
        let shade = if inside { 90 } else { 35 };
        texel.copy_from_slice(&[shade, shade + 4, shade + 10, 255]);
    }
    px
}

fn main() -> Result<(), Box<dyn Error>> {
    let frame = overhead_frame();
    let mut texture = None;
    let mut homography: Option<Homography> = None;

    common::run(
        "fathom - homography overlay",
        Meters(1.0),
        move |ctx, input| {
            if texture.is_none() {
                if let (Some(w), Some(h)) = (NonZeroU32::new(W), NonZeroU32::new(H)) {
                    texture =
                        upload_texture(ctx, &frame, w, h, Format::Rgba8, Filter::Nearest).ok();
                }
            }

            let mut f = begin_frame(ctx);
            let Some(tex) = &texture else {
                f.end();
                return;
            };
            let view = f.viewport().inset(24.0).fit_aspect(tex.aspect());
            draw_texture(&mut f, tex, view, Color::WHITE);
            draw_bbox(&mut f, view, Color::rgb(80, 80, 90));

            // Clicks arrive in window pixels; the homography wants image pixels.
            let to_image = |p: ImagePoint| {
                ImagePoint::new(
                    (p.0.x - view.x) / view.w * W as f32,
                    (p.0.y - view.y) / view.h * H as f32,
                )
            };
            let to_window = |p: ImagePoint| {
                ImagePoint::new(
                    view.x + p.0.x / W as f32 * view.w,
                    view.y + p.0.y / H as f32 * view.h,
                )
            };

            if homography.is_none() && input.clicks.len() >= 4 {
                // A one-metre square of workcell floor, in plane coordinates.
                let plane = [
                    Vec2::new(0.0, 0.0),
                    Vec2::new(1.0, 0.0),
                    Vec2::new(1.0, 1.0),
                    Vec2::new(0.0, 1.0),
                ];
                let image: Vec<Vec2> = input
                    .clicks
                    .iter()
                    .take(4)
                    .map(|&c| to_image(c).0)
                    .collect();
                homography = homography_from_correspondences(&plane, &image).ok();
            }

            for (i, &c) in input.clicks.iter().take(4).enumerate() {
                draw_bbox(
                    &mut f,
                    Rect::new(c.0.x - 4.0, c.0.y - 4.0, 8.0, 8.0),
                    Color::YELLOW,
                );
                draw_text_at(
                    &mut f,
                    ImagePoint::new(c.0.x + 8.0, c.0.y - 4.0),
                    &format!("{i}"),
                    TextScale::X1,
                    Color::YELLOW,
                );
            }

            if let Some(h) = &homography {
                // The floor grid, warped. Ten lines each way, in metres.
                let mut strip: Vec<ImagePoint> = Vec::with_capacity(11);
                for k in 0..=10 {
                    let t = k as f32 / 10.0;
                    for line in [[(t, 0.0), (t, 1.0)], [(0.0, t), (1.0, t)]] {
                        strip.clear();
                        for step in 0..=10 {
                            let s = step as f32 / 10.0;
                            let (a, b) = (line[0], line[1]);
                            let p = PlanePoint::new(a.0 + (b.0 - a.0) * s, a.1 + (b.1 - a.1) * s);
                            strip.push(to_window(warp(p, h)));
                        }
                        draw_line_strip_2d(&mut f, &strip, Color::GREEN.with_alpha(0.7));
                    }
                }

                // And the inverse, live: where on the floor is the cursor?
                if view.contains(input.cursor.0.x, input.cursor.0.y) {
                    let on_floor = unwarp(to_image(input.cursor), h);
                    draw_text_at(
                        &mut f,
                        ImagePoint::new(view.x, view.bottom() + 10.0),
                        &format!("floor: ({:+.3}, {:+.3}) m", on_floor.0.x, on_floor.0.y),
                        TextScale::X2,
                        Color::WHITE,
                    );
                }
            } else {
                draw_text_at(
                    &mut f,
                    ImagePoint::new(view.x, view.y - 16.0),
                    "click the 4 corners of the light quad: TL, TR, BR, BL",
                    TextScale::X1,
                    Color::WHITE,
                );
            }
            f.end();
        },
    )
}
