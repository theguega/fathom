//! Transforms straight from the control stack, no FK inside fathom.
//!
//! Your controller already computes link transforms every tick. Having the
//! viewer recompute them duplicates the truth and invites version skew between
//! what the controller thinks the arm is doing and what the screen shows, so
//! fathom takes `&[Mat4]` and draws it.
//!
//! Run with `cargo run -p fathom --example robot_arm`.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
//  ^ example code: every cast here is small and deliberate

mod common;

use std::error::Error;

use fathom::prelude::*;

/// Stand-in for `robot.left.link_transforms()`.
fn arm(t: f32, side: f32) -> [Mat4; 7] {
    let mut m = Mat4::from_translation(Vec3::new(side * 0.25, 0.0, 0.0));
    core::array::from_fn(|i| {
        let axis = if i % 2 == 0 { Vec3::Y } else { Vec3::X };
        let q = (t * (1.0 + i as f32 * 0.2) + side).sin() * 0.5;
        m *= Mat4::from_translation(Vec3::new(0.0, 0.14, 0.0)) * Mat4::from_axis_angle(axis, q);
        m
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut executed: Vec<WorldPoint> = Vec::with_capacity(600);

    common::run("fathom - robot arm", Meters(2.2), move |ctx, input| {
        let left = arm(input.time, -1.0);
        let right = arm(input.time * 0.8, 1.0);

        // The end-effector is the last link's origin. Caller-owned history.
        if let Some(tip) = left
            .last()
            .map(|m| WorldPoint::from_repr(m.w_axis.truncate()))
        {
            if executed.len() == 600 {
                executed.remove(0);
            }
            executed.push(tip);
        }

        // A predicted action chunk, as a policy would emit it.
        let predicted: Vec<WorldPoint> = (0..20)
            .map(|k| {
                let m = arm(input.time + k as f32 * 0.05, -1.0);
                WorldPoint::from_repr(m.last().map_or(Vec3::ZERO, |m| m.w_axis.truncate()))
            })
            .collect();

        let mut f = begin_frame(ctx);
        let mut s = f.scene(&input.orbit.camera(ctx_aspect(&f)));

        draw_grid(&mut s, 24, Meters(0.1), Color::rgb(42, 42, 52));
        draw_frames(&mut s, &left, Meters(0.05));
        draw_frames(&mut s, &right, Meters(0.05));
        draw_line_strip_3d(&mut s, &executed, Color::GREEN);
        draw_line_strip_3d(&mut s, &predicted, Color::RED.with_alpha(0.5));
        s.end();

        draw_text_at(
            &mut f,
            ImagePoint::new(12.0, 12.0),
            "green: executed    red: predicted chunk    drag to orbit",
            TextScale::X1,
            Color::GRAY,
        );
        f.end();
    })
}

fn ctx_aspect(f: &Frame<'_>) -> f32 {
    let v = f.viewport();
    v.w / v.h
}
