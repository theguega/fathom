//! A heatmap is a texture. There is no heatmap primitive.
//!
//! `NxN` distances go through `colormap` to `upload_texture` to `draw_texture`.
//! That is the whole path, and it is the same path a depth map or an attention
//! map takes.
//!
//! Run with `cargo run -p fathom --example similarity_matrix`.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
//  ^ example code: every cast here is small and deliberate

mod common;

use std::{error::Error, num::NonZeroU32};

use fathom::prelude::*;

const N: usize = 64;

fn main() -> Result<(), Box<dyn Error>> {
    // Sixty-four embeddings; three loose clusters.
    let embeddings: Vec<[f32; 8]> = (0..N)
        .map(|i| {
            let cluster = (i / 22) as f32;
            core::array::from_fn(|k| {
                ((i as f32 * 0.3 + k as f32).sin()).mul_add(0.25, cluster + k as f32 * 0.05)
            })
        })
        .collect();

    // Cosine similarity, in caller code. fathom never sees the metric.
    let mut similarity = vec![0.0f32; N * N];
    for (i, a) in embeddings.iter().enumerate() {
        for (j, b) in embeddings.iter().enumerate() {
            let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            if let Some(cell) = similarity.get_mut(i * N + j) {
                *cell = dot / (na * nb).max(1e-6);
            }
        }
    }

    let pixels = colormap(&similarity, 0.0..1.0, ColorMap::Viridis);
    let mut texture = None;

    common::run(
        "fathom - similarity matrix",
        Meters(1.0),
        move |ctx, input| {
            // Textures are allocated between frames, never inside one.
            if texture.is_none() {
                if let (Some(w), Some(h)) = (NonZeroU32::new(N as u32), NonZeroU32::new(N as u32)) {
                    texture = upload_texture(
                        ctx,
                        bytemuck_cast(&pixels),
                        w,
                        h,
                        Format::Rgba8,
                        Filter::Nearest,
                    )
                    .ok();
                }
            }

            let mut f = begin_frame(ctx);
            if let Some(tex) = &texture {
                let plot = f.viewport().inset(48.0).fit_aspect(1.0);
                draw_texture(&mut f, tex, plot, Color::WHITE);
                draw_bbox(&mut f, plot, Color::rgb(90, 90, 100));

                // Read a cell out under the cursor: `Rect` maths, in caller code.
                let c = input.cursor;
                if plot.contains(c.0.x, c.0.y) {
                    let i = ((c.0.y - plot.y) / plot.h * N as f32) as usize;
                    let j = ((c.0.x - plot.x) / plot.w * N as f32) as usize;
                    let v = similarity.get(i * N + j).copied().unwrap_or(0.0);
                    draw_text_at(
                        &mut f,
                        ImagePoint::new(plot.x, plot.bottom() + 10.0),
                        &format!("[{i:>3}, {j:>3}] = {v:.4}"),
                        TextScale::X2,
                        Color::WHITE,
                    );
                }
            }
            draw_text_at(
                &mut f,
                ImagePoint::new(12.0, 12.0),
                "64x64 cosine similarity, Viridis. Hover to read a cell.",
                TextScale::X1,
                Color::GRAY,
            );
            f.end();
        },
    )
}

/// `Color` is `#[repr(transparent)]` over `u32`, so this is a view, not a copy.
fn bytemuck_cast(colors: &[Color]) -> &[u8] {
    // SAFETY-free: `Color` is `Pod`, so bytemuck checks this at compile time.
    bytemuck::cast_slice(colors)
}
