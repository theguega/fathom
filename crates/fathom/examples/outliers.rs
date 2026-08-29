//! The whole "outlier analysis" story, in caller code, with no analysis crate.
//!
//! fathom knows nothing about outliers. The caller computes a score per demo
//! and maps it through `colormap`; the drawing is one call. Swap the metric for
//! cosine similarity and this is the similarity example. Slice `demos` by
//! subtask boundary first and it is the granularity example. The examples vary,
//! the primitives do not.
//!
//! Run with `cargo run -p fathom --example outliers`.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
//  ^ example code: every cast here is small and deliberate

mod common;

use std::error::Error;

use fathom::prelude::*;

/// A cheap stand-in for DTW or any other trajectory distance. Yours goes here.
fn distance(a: &[WorldPoint], b: &[WorldPoint]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(p, q)| (p.0 - q.0).length())
        .sum::<f32>()
        / a.len().max(1) as f32
}

/// Twenty demonstrations of the same reach, some of them sloppy.
fn demos() -> Vec<Vec<WorldPoint>> {
    (0..20)
        .map(|d| {
            let wobble = if d % 7 == 0 { 0.25 } else { 0.03 };
            let phase = d as f32 * 0.7;
            (0..120)
                .map(|i| {
                    let t = i as f32 / 119.0;
                    WorldPoint::new(
                        t.mul_add(1.2, -0.6),
                        (t * 3.0 + phase).sin() * wobble + 0.3,
                        (t * 5.0 + phase).cos() * wobble,
                    )
                })
                .collect()
        })
        .collect()
}

fn main() -> Result<(), Box<dyn Error>> {
    let demos = demos();
    let reference: Vec<WorldPoint> = (0..120)
        .map(|i| {
            let t = i as f32 / 119.0;
            WorldPoint::new(t.mul_add(1.2, -0.6), 0.3, 0.0)
        })
        .collect();

    // The caller computes the scores; fathom never sees them.
    let scores: Vec<f32> = demos.iter().map(|d| distance(d, &reference)).collect();
    let max = scores.iter().copied().fold(0.0f32, f32::max);

    // Scratch, filled once: `colormap_into` is the form used in a loop.
    let mut colors = vec![Color::TRANSPARENT; scores.len()];
    colormap_into(&scores, 0.0..max, ColorMap::Turbo, &mut colors);

    common::run("fathom - outliers", Meters(2.0), move |ctx, input| {
        let mut f = begin_frame(ctx);
        let v = f.viewport();
        let mut s = f.scene(&input.orbit.camera(v.w / v.h));

        draw_grid(&mut s, 20, Meters(0.1), Color::rgb(40, 40, 50));
        for (demo, &c) in demos.iter().zip(&colors) {
            draw_line_strip_3d(&mut s, demo, c.with_alpha(0.55));
        }
        draw_line_strip_3d(&mut s, &reference, Color::WHITE);
        s.end();

        draw_text_at(
            &mut f,
            ImagePoint::new(12.0, 12.0),
            &format!(
                "{} demos, coloured by distance from the reference (Turbo)",
                demos.len()
            ),
            TextScale::X1,
            Color::GRAY,
        );
        f.end();
    })
}
