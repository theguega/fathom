//! The integration contract, in one file.
//!
//! This is what an MCAP, LeRobot or ROS bag adapter needs to expose, and all it
//! needs to expose: slices of `fathom-core` types. No `Source` trait, no plugin
//! registry, no dynamic dispatch. An adapter is just a crate whose functions
//! happen to return types fathom already understands, which is why one can be
//! written against `fathom-core` alone, without ever depending on the renderer.
//!
//! Run with `cargo run -p fathom --example adapter_stub`.
#![allow(clippy::cast_precision_loss)] // example code: every cast here is small and deliberate

use std::error::Error;

use fathom::prelude::*;

/// One episode, exactly as an adapter would hand it over.
///
/// Note what is *not* here: no decode state, no seek cursor, no callbacks.
/// Streams are timestamped slices, and every stream has its own rate.
struct Episode {
    instruction: String,
    intrinsics: Intrinsics,
    /// 30Hz camera, as RGBA8 ready for `upload_texture`.
    video: Vec<(Timestamp, Vec<u8>)>,
    /// 500Hz joint states, off the control loop.
    joints: Vec<(Timestamp, [f32; 7])>,
    /// The end-effector path, already in world coordinates.
    ee_path: Vec<(Timestamp, WorldPoint)>,
}

/// Stand-in for `mcap::read_episode(path)` or `lerobot::load(repo_id, episode)`.
fn load_episode() -> Result<Episode, CalibError> {
    let (w, h) = (32usize, 24usize);

    let video = (0..30)
        .map(|i| {
            let t = Timestamp(i * 33_333_333);
            let mut rgba = vec![0u8; w * h * 4];
            for (p, texel) in rgba.chunks_exact_mut(4).enumerate() {
                let x = u8::try_from((p % w) * 8 % 256).unwrap_or(0);
                texel.copy_from_slice(&[x, u8::try_from(i * 8).unwrap_or(0), 128, 255]);
            }
            (t, rgba)
        })
        .collect();

    let joints = (0..500)
        .map(|i| {
            let t = Timestamp(i * 2_000_000);
            let phase = f32::from(u16::try_from(i).unwrap_or(0)) * 0.01;
            (
                t,
                core::array::from_fn(|j| phase.sin() * (1.0 + j as f32 * 0.1)),
            )
        })
        .collect();

    let ee_path = (0..500)
        .map(|i| {
            let t = Timestamp(i * 2_000_000);
            let phase = f32::from(u16::try_from(i).unwrap_or(0)) * 0.01;
            (
                t,
                WorldPoint::new(phase.cos() * 0.3, phase * 0.001, phase.sin() * 0.3),
            )
        })
        .collect();

    Ok(Episode {
        instruction: "pick up the red block".to_owned(),
        intrinsics: Intrinsics::new(600.0, 600.0, 320.0, 240.0)?,
        video,
        joints,
        ee_path,
    })
}

/// Aligning streams is caller code, and it is one line.
///
/// Timestamps, not frame indices, are the shared axis: video arrives at 30Hz,
/// joint states at 500Hz, the instruction once per episode. A frame index
/// cannot express that, so fathom never offers one.
fn sample_at<T>(stream: &[(Timestamp, T)], now: Timestamp) -> Option<&T> {
    let i = stream.partition_point(|(t, _)| *t <= now);
    stream.get(i.checked_sub(1)?).map(|(_, v)| v)
}

fn main() -> Result<(), Box<dyn Error>> {
    let ep = load_episode()?;

    println!("instruction: {:?}", ep.instruction);
    println!(
        "intrinsics:  fx={} cx={}",
        ep.intrinsics.fx(),
        ep.intrinsics.cx()
    );
    println!(
        "streams:     {} video frames, {} joint states, {} ee samples",
        ep.video.len(),
        ep.joints.len(),
        ep.ee_path.len()
    );

    // Scrubbing is the caller choosing which slice to hand to the draw calls.
    // In live mode there is nothing to seek; in playback the adapter seeks.
    // Same draw code either way, which is the fidelity test.
    for ms in [0, 250, 500, 999] {
        let now = Timestamp(ms * 1_000_000);
        let frame = sample_at(&ep.video, now).map_or(0, Vec::len);
        let joint0 = sample_at(&ep.joints, now).map_or(0.0, |q| q[0]);
        let ee = sample_at(&ep.ee_path, now)
            .copied()
            .unwrap_or(WorldPoint::ORIGIN);
        println!(
            "t={:>4}ms  video={frame:>4}B  joint0={joint0:+.3}  ee=({:+.3}, {:+.3}, {:+.3})",
            ms, ee.0.x, ee.0.y, ee.0.z
        );
    }

    // The path is already a `&[WorldPoint]`, so it feeds `draw_line_strip_3d`
    // directly. That is the entire handoff: no conversion, no adapter trait.
    let path: Vec<WorldPoint> = ep.ee_path.iter().map(|(_, p)| *p).collect();
    println!(
        "\nready to draw: {} points, no conversion needed",
        path.len()
    );

    Ok(())
}
