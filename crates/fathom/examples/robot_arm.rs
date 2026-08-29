//! A bimanual cell: two 6-DOF arms with parallel grippers, drawn from the link
//! transforms your controller already computed.
//!
//! There is no forward kinematics in fathom, and no robot model. Your control
//! stack has these transforms every tick; having the viewer recompute them
//! duplicates the truth and invites skew between what the controller thinks the
//! arm is doing and what the screen shows. So the viewer takes `&[Mat4]`.
//!
//! The proportions here are those of a YAM-class arm - six revolute joints, a
//! two-finger parallel gripper, roughly 60cm of reach - because that is the
//! shape the primitives have to look right on, not because fathom knows
//! anything about it.
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

/// Joints per arm, excluding the gripper.
const DOF: usize = 6;

/// One revolute joint: where it sits relative to its parent, and its axis.
struct Joint {
    offset: Vec3,
    axis: Vec3,
}

/// A six-axis arm: shoulder yaw and pitch, elbow, then a three-axis wrist.
fn chain() -> [Joint; DOF] {
    [
        Joint {
            offset: Vec3::new(0.00, 0.00, 0.07),
            axis: Vec3::Z,
        },
        Joint {
            offset: Vec3::new(0.02, 0.03, 0.05),
            axis: Vec3::Y,
        },
        Joint {
            offset: Vec3::new(0.28, 0.00, 0.00),
            axis: Vec3::NEG_Y,
        },
        Joint {
            offset: Vec3::new(0.24, 0.00, 0.06),
            axis: Vec3::Y,
        },
        Joint {
            offset: Vec3::new(0.07, 0.03, 0.04),
            axis: Vec3::Z,
        },
        Joint {
            offset: Vec3::new(0.04, 0.00, -0.05),
            axis: Vec3::X,
        },
    ]
}

/// Forward kinematics: an accumulate-and-collect, in the caller's code.
fn fk(base: Mat4, joints: &[f32; DOF], chain: &[Joint; DOF], out: &mut Vec<Mat4>) {
    out.clear();
    let mut t = base;
    for (q, link) in joints.iter().zip(chain) {
        t *= Mat4::from_translation(link.offset) * Mat4::from_axis_angle(link.axis, *q);
        out.push(t);
    }
}

/// The tool centre point: a fixed offset past the last wrist joint.
fn tcp(links: &[Mat4]) -> Mat4 {
    links.last().copied().unwrap_or(Mat4::IDENTITY)
        * Mat4::from_translation(Vec3::new(0.16, 0.0, 0.004))
}

/// Draw a two-finger parallel gripper at the flange. `opening` is in metres,
/// which is what the prismatic joints report.
fn draw_gripper(s: &mut Scene<'_, '_>, flange: Mat4, opening: Meters, color: Color) {
    let half = opening.get() * 0.5;
    for side in [-1.0, 1.0] {
        // Each finger: out to its rail, then forward to the tip.
        let root = flange * Vec3::new(0.02, side * half, 0.0).extend(1.0);
        let tip = flange * Vec3::new(0.09, side * half, 0.0).extend(1.0);
        draw_line_3d(
            s,
            WorldPoint::from_repr(root.truncate()),
            WorldPoint::from_repr(tip.truncate()),
            color,
        );
        let base = flange * Vec3::new(0.02, 0.0, 0.0).extend(1.0);
        draw_line_3d(
            s,
            WorldPoint::from_repr(base.truncate()),
            WorldPoint::from_repr(root.truncate()),
            color,
        );
    }
}

/// Which of the cell's control modes is driving the arms right now.
///
/// This is a HUD string, nothing more: fathom has no notion of a mode, and the
/// orchestration layer that does is yours.
#[derive(Clone, Copy)]
enum Mode {
    Autonomy,
    Scripted,
    Teleop,
    Nudge,
}

impl Mode {
    const fn label(self) -> &'static str {
        match self {
            Self::Autonomy => "AUTONOMY   policy in the loop",
            Self::Scripted => "SCRIPTED   waypoint controller",
            Self::Teleop => "TELEOP     leader arms, absolute",
            Self::Nudge => "NUDGE      HID, relative offsets",
        }
    }
    const fn color(self) -> Color {
        match self {
            Self::Autonomy => Color::rgb(80, 220, 140),
            Self::Scripted => Color::rgb(120, 170, 255),
            Self::Teleop => Color::rgb(255, 190, 80),
            Self::Nudge => Color::rgb(220, 140, 255),
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let chain = chain();
    // The two arms are mounted 50cm apart, facing into the workspace.
    let bases = [
        Mat4::from_translation(Vec3::new(0.0, 0.0, -0.25)),
        Mat4::from_translation(Vec3::new(0.0, 0.0, 0.25))
            * Mat4::from_rotation_y(core::f32::consts::PI),
    ];

    let mut links = [Vec::with_capacity(DOF), Vec::with_capacity(DOF)];
    let mut executed: [Vec<WorldPoint>; 2] = [Vec::with_capacity(600), Vec::with_capacity(600)];
    let mut predicted: Vec<WorldPoint> = Vec::with_capacity(24);

    common::run("fathom - bimanual cell", Meters(2.0), move |ctx, input| {
        let t = input.time;
        // Stand-in for the control loop. Yours reports real joint angles.
        let angles = |arm: usize, ahead: f32| -> [f32; DOF] {
            core::array::from_fn(|j| {
                ((t + ahead) * (0.5 + j as f32 * 0.13) + arm as f32 * 1.7).sin() * 0.5
            })
        };
        let opening = |arm: usize| Meters(((t * 0.9 + arm as f32).sin() * 0.5 + 0.5) * 0.047);

        // Cycle the modes so the HUD has something to show.
        let mode = match ((t * 0.25) as u32) % 4 {
            0 => Mode::Autonomy,
            1 => Mode::Scripted,
            2 => Mode::Teleop,
            _ => Mode::Nudge,
        };

        for (arm, ((base, links), executed)) in
            bases.iter().zip(&mut links).zip(&mut executed).enumerate()
        {
            fk(*base, &angles(arm, 0.0), &chain, links);
            let tip = WorldPoint::from_repr(tcp(links).w_axis.truncate());
            if executed.len() == 600 {
                executed.remove(0);
            }
            executed.push(tip);
        }

        // The action chunk a policy would emit: the next 24 steps, right arm.
        predicted.clear();
        let mut scratch = Vec::with_capacity(DOF);
        let base = bases.first().copied().unwrap_or(Mat4::IDENTITY);
        for k in 0..24 {
            fk(base, &angles(0, k as f32 * 0.06), &chain, &mut scratch);
            predicted.push(WorldPoint::from_repr(tcp(&scratch).w_axis.truncate()));
        }

        let mut f = begin_frame(ctx);
        let v = f.viewport();
        let mut s = f.scene(&input.orbit.camera(v.w / v.h));

        draw_grid(&mut s, 24, Meters(0.1), Color::rgb(42, 42, 52));
        for (arm, (links, executed)) in links.iter().zip(&executed).enumerate() {
            draw_frames(&mut s, links, Meters(0.05));
            draw_gripper(&mut s, tcp(links), opening(arm), Color::rgb(230, 230, 240));
            draw_line_strip_3d(&mut s, executed, Color::GREEN.with_alpha(0.9));
        }
        draw_line_strip_3d(&mut s, &predicted, Color::RED.with_alpha(0.6));
        s.end();

        // The HUD. Layout is Rect maths; there is no panel manager.
        draw_text_at(
            &mut f,
            ImagePoint::new(14.0, 14.0),
            mode.label(),
            TextScale::X2,
            mode.color(),
        );
        draw_text_at(
            &mut f,
            ImagePoint::new(14.0, 38.0),
            "green: executed   red: predicted chunk   drag to orbit, wheel to zoom",
            TextScale::X1,
            Color::GRAY,
        );
        for (arm, name) in ["left ", "right"].into_iter().enumerate() {
            draw_text_at(
                &mut f,
                ImagePoint::new(14.0, 58.0 + arm as f32 * 12.0),
                &format!("{name} gripper {:>5.1} mm", opening(arm).get() * 1000.0),
                TextScale::X1,
                Color::rgb(150, 150, 160),
            );
        }
        f.end();
    })
}
