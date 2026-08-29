//! A depth buffer as false colour, and `unproject` on hover to read metres.
//!
//! Note what the example does *not* do: it never asks fathom to turn depth into
//! points. `depth_to_points` is a data transform, and it belongs to the adapter
//! that already knows the encoding, the camera model and whether the stream is
//! rectified. Iterating undistortion across 307k pixels would also eat the
//! entire frame budget, where doing it for one hovered pixel is free.
//!
//! Run with `cargo run -p fathom --example depth_probe`.
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

fn main() -> Result<(), Box<dyn Error>> {
    let k = Intrinsics::new(320.0, 320.0, 160.0, 120.0)?
        .with_brown_conrady([-0.28, 0.07, 0.0], [0.0, 0.0])?;
    let e = Extrinsics::IDENTITY;

    // A synthetic depth map in metres: a tilted floor with a box on it.
    let depth: Vec<f32> = (0..W * H)
        .map(|i| {
            let (x, y) = ((i % W) as f32, (i / W) as f32);
            let floor = 1.2 + (H as f32 - y) * 0.012;
            let in_box = (110.0..210.0).contains(&x) && (90.0..170.0).contains(&y);
            if in_box { floor - 0.35 } else { floor }
        })
        .collect();

    // Depth becomes RGBA the same way every other scalar field does.
    let colors = colormap(&depth, 1.0..2.5, ColorMap::Turbo);
    let mut texture = None;

    common::run("fathom - depth probe", Meters(1.0), move |ctx, input| {
        if texture.is_none() {
            if let (Some(w), Some(h)) = (NonZeroU32::new(W), NonZeroU32::new(H)) {
                texture = upload_texture(
                    ctx,
                    bytemuck::cast_slice(&colors),
                    w,
                    h,
                    Format::Rgba8,
                    Filter::Nearest,
                )
                .ok();
            }
        }

        let mut f = begin_frame(ctx);
        let Some(tex) = &texture else {
            f.end();
            return;
        };
        let view = f.viewport().inset(32.0).fit_aspect(tex.aspect());
        draw_texture(&mut f, tex, view, Color::WHITE);
        draw_bbox(&mut f, view, Color::rgb(80, 80, 90));

        let c = input.cursor;
        if view.contains(c.0.x, c.0.y) {
            // Window pixel -> image pixel -> metric world point.
            let px = ImagePoint::new(
                (c.0.x - view.x) / view.w * W as f32,
                (c.0.y - view.y) / view.h * H as f32,
            );
            let idx = (px.0.y as usize) * W as usize + px.0.x as usize;
            let z = depth.get(idx).copied().unwrap_or(0.0);
            let world = unproject(px, Meters(z), &k, &e);

            draw_bbox(
                &mut f,
                Rect::new(c.0.x - 5.0, c.0.y - 5.0, 10.0, 10.0),
                Color::WHITE,
            );
            draw_text_at(
                &mut f,
                ImagePoint::new(view.x, view.bottom() + 10.0),
                &format!(
                    "px ({:>3.0}, {:>3.0})   depth {z:.3} m   world ({:+.3}, {:+.3}, {:+.3})",
                    px.0.x, px.0.y, world.0.x, world.0.y, world.0.z
                ),
                TextScale::X2,
                Color::WHITE,
            );

            // The round trip, drawn: project it back and mark the pixel.
            if let Some(back) = project(world, &k, &e) {
                let mark = ImagePoint::new(
                    view.x + back.0.x / W as f32 * view.w,
                    view.y + back.0.y / H as f32 * view.h,
                );
                draw_line_2d(&mut f, c, mark, Color::GREEN);
            }
        }

        draw_text_at(
            &mut f,
            ImagePoint::new(12.0, 12.0),
            "depth via colormap -> texture. Hover to unproject a pixel to metres.",
            TextScale::X1,
            Color::GRAY,
        );
        f.end();
    })
}
