//! Why fathom needs no kinematics crate.
//!
//! A serial chain is an accumulate-and-collect, in the caller's code, with
//! glam already on hand. Nine lines, and it feeds `draw_frames` directly.
//! Anyone needing URDF parsing, IK or dynamics reaches for `k` or `kinetix` in
//! their own crate and passes the result in. That boundary is the point.
//!
//! Run with `cargo run -p fathom --example fk_minimal`.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
//  ^ example code: every cast here is small and deliberate

mod common;

use std::error::Error;

use fathom::prelude::*;

/// One joint: where it sits relative to its parent, and what it rotates about.
struct Link {
    fixed: Mat4,
    axis: Vec3,
}

/// Forward kinematics, entire.
fn fk(joints: &[f32], chain: &[Link], out: &mut Vec<Mat4>) {
    out.clear();
    let mut t = Mat4::IDENTITY;
    for (q, link) in joints.iter().zip(chain) {
        t *= link.fixed * Mat4::from_axis_angle(link.axis, *q);
        out.push(t);
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    // A WAM-ish 7-DOF chain: alternating axes, 15cm links.
    let chain: Vec<Link> = (0..7)
        .map(|i| Link {
            fixed: Mat4::from_translation(Vec3::new(0.0, 0.15, 0.0)),
            axis: if i % 2 == 0 { Vec3::Y } else { Vec3::X },
        })
        .collect();

    let mut links: Vec<Mat4> = Vec::with_capacity(chain.len());

    common::run("fathom - fk_minimal", Meters(2.0), move |ctx, input| {
        let joints: Vec<f32> = (0..7)
            .map(|i| (input.time * (0.6 + i as f32 * 0.15)).sin() * 0.6)
            .collect();
        fk(&joints, &chain, &mut links);

        let mut f = begin_frame(ctx);
        let v = f.viewport();
        let mut s = f.scene(&input.orbit.camera(v.w / v.h));
        draw_grid(&mut s, 20, Meters(0.1), Color::rgb(42, 42, 52));
        draw_frames(&mut s, &links, Meters(0.06));
        s.end();

        for (i, q) in joints.iter().enumerate() {
            draw_text_at(
                &mut f,
                ImagePoint::new(12.0, 12.0 + i as f32 * 12.0),
                &format!("q{i} {q:+.3} rad"),
                TextScale::X1,
                Color::rgb(150, 150, 160),
            );
        }
        f.end();
    })
}
