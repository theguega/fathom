//! Why fathom needs no kinematics crate.
//!
//! A serial chain is an accumulate-and-collect, in the caller's code, with
//! glam already on hand. Nine lines, and it feeds `draw_frames` directly.
//! Anyone needing URDF parsing, IK or dynamics reaches for `k` or `kinetix` in
//! their own crate and passes the result in. That boundary is the point.
//!
//! Six revolute joints and a two-finger gripper, which is the shape of the arms
//! this was built against. Read the offsets and axes out of your URDF once at
//! startup and this loop is the whole of your viewer-side kinematics.
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
    // A six-axis arm: shoulder yaw and pitch, elbow, then a three-axis wrist.
    // These are the numbers you would read out of a URDF's <origin> and <axis>.
    let chain = [
        Link {
            fixed: Mat4::from_translation(Vec3::new(0.00, 0.00, 0.07)),
            axis: Vec3::Z,
        },
        Link {
            fixed: Mat4::from_translation(Vec3::new(0.02, 0.03, 0.05)),
            axis: Vec3::Y,
        },
        Link {
            fixed: Mat4::from_translation(Vec3::new(0.28, 0.00, 0.00)),
            axis: Vec3::NEG_Y,
        },
        Link {
            fixed: Mat4::from_translation(Vec3::new(0.24, 0.00, 0.06)),
            axis: Vec3::Y,
        },
        Link {
            fixed: Mat4::from_translation(Vec3::new(0.07, 0.03, 0.04)),
            axis: Vec3::Z,
        },
        Link {
            fixed: Mat4::from_translation(Vec3::new(0.04, 0.00, -0.05)),
            axis: Vec3::X,
        },
    ];

    let mut links: Vec<Mat4> = Vec::with_capacity(chain.len());

    common::run("fathom - fk_minimal", Meters(2.0), move |ctx, input| {
        let joints: Vec<f32> = (0..chain.len())
            .map(|i| (input.time * (0.5 + i as f32 * 0.13)).sin() * 0.5)
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
