//! Spatial variance: overplot at low alpha, plus an ellipsoid per waypoint.
//!
//! The covariance is computed in the caller's code, from the caller's own
//! statistics. `draw_wire_ellipsoid` takes the matrix that maps the unit sphere
//! to the shell you want, which for a 1-sigma volume is the Cholesky factor.
//!
//! Run with `cargo run -p fathom --example variance`.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
//  ^ example code: every cast here is small and deliberate

mod common;

use std::error::Error;

use fathom::prelude::*;

const DEMOS: usize = 24;
const STEPS: usize = 60;

/// Cholesky factor of a symmetric positive-definite 3x3, so `L * L^T = cov`.
/// Twenty lines of caller code, and it is why no stats crate is a dependency.
#[allow(clippy::indexing_slicing, clippy::needless_range_loop)] // fixed 3x3, every index bounded by a constant loop
fn cholesky(cov: Mat3) -> Mat3 {
    let a = cov.to_cols_array_2d();
    let mut l = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..=i {
            let mut sum = a[j][i];
            for k in 0..j {
                sum -= l[k][i] * l[k][j];
            }
            if i == j {
                l[j][i] = sum.max(1e-12).sqrt();
            } else {
                l[j][i] = sum / l[j][j].max(1e-12);
            }
        }
    }
    Mat3::from_cols_array_2d(&l)
}

fn main() -> Result<(), Box<dyn Error>> {
    // Twenty-four noisy executions of one motion.
    let demos: Vec<Vec<WorldPoint>> = (0..DEMOS)
        .map(|d| {
            let phase = d as f32 * 0.9;
            (0..STEPS)
                .map(|i| {
                    let t = i as f32 / (STEPS - 1) as f32;
                    // Noise that grows through the reach, as a real policy's does.
                    let spread = t * 0.09;
                    WorldPoint::new(
                        t.mul_add(1.0, -0.5) + (phase * 1.7).sin() * spread,
                        (t * core::f32::consts::PI).sin() * 0.4 + (phase * 2.3).cos() * spread,
                        (phase * 3.1).sin() * spread,
                    )
                })
                .collect()
        })
        .collect();

    // Mean and covariance per waypoint: the caller's statistics, once.
    let mut means = Vec::with_capacity(STEPS);
    let mut shells = Vec::with_capacity(STEPS);
    for i in 0..STEPS {
        let pts: Vec<Vec3> = demos.iter().filter_map(|d| d.get(i)).map(|p| p.0).collect();
        let n = pts.len().max(1) as f32;
        let mean = pts.iter().copied().sum::<Vec3>() / n;

        let mut cov = Mat3::ZERO;
        for p in &pts {
            let d = *p - mean;
            cov += Mat3::from_cols(d * d.x, d * d.y, d * d.z);
        }
        cov *= 1.0 / n;

        means.push(WorldPoint::from_repr(mean));
        shells.push(cholesky(cov));
    }

    common::run("fathom - variance", Meters(2.0), move |ctx, input| {
        let mut f = begin_frame(ctx);
        let v = f.viewport();
        let mut s = f.scene(&input.orbit.camera(v.w / v.h));

        draw_grid(&mut s, 20, Meters(0.1), Color::rgb(40, 40, 50));

        // Overplot: twenty-four passes at low alpha read as a density.
        for demo in &demos {
            draw_line_strip_3d(&mut s, demo, Color::CYAN.with_alpha(0.18));
        }
        draw_line_strip_3d(&mut s, &means, Color::WHITE);

        // One 2-sigma shell every fifth waypoint, so the volume stays readable.
        for (mean, shell) in means.iter().zip(&shells).step_by(5) {
            draw_wire_ellipsoid(&mut s, *mean, *shell * 2.0, Color::YELLOW.with_alpha(0.6));
        }
        s.end();

        draw_text_at(
            &mut f,
            ImagePoint::new(12.0, 12.0),
            "24 demos at alpha 0.18, mean in white, 2-sigma shells in yellow",
            TextScale::X1,
            Color::GRAY,
        );
        f.end();
    })
}
