//! The fidelity test: identical draw calls, different init.
//!
//! If the draw code here differs from `live_viewer`, the API leaked a display
//! or a liveness assumption. It does not: the only difference is `Ctx::headless`
//! instead of `Ctx::new`, and `read_pixels` instead of a window.
//!
//! With `--features media` it writes `out.mp4` directly. Without it, it writes
//! raw RGBA8 frames, top row first, which is what any encoder wants on stdin:
//!
//! ```sh
//! cargo run -p fathom --example headless_export --features media
//! # or, without the feature:
//! cargo run -p fathom --example headless_export
//! ffmpeg -f rawvideo -pix_fmt rgba -s 1280x720 -r 30 -i frames.rgba out.mp4
//! ```
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
//  ^ example code: every cast here is small and deliberate

use std::{error::Error, num::NonZeroU32};

use fathom::prelude::*;

/// The only thing that differs between the two builds: where frames go.
#[allow(unreachable_pub)] // an example is its own crate root
#[cfg(feature = "media")]
mod sink {
    use fathom::media::{EncodeOptions, Encoder, Open};

    pub struct Sink(Encoder<Open>);

    impl Sink {
        pub fn new(w: u32, h: u32) -> Result<Self, Box<dyn std::error::Error>> {
            let options = EncodeOptions::new(30);
            Ok(Self(Encoder::new("out.mp4", w, h, &options)?))
        }
        pub fn push(&mut self, rgba: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
            self.0.write(rgba)?;
            Ok(())
        }
        pub fn close(self) -> Result<String, Box<dyn std::error::Error>> {
            let done = self.0.finish()?;
            Ok(format!("out.mp4 ({} frames)", done.frames()))
        }
    }
}

#[allow(unreachable_pub)] // an example is its own crate root
#[cfg(not(feature = "media"))]
mod sink {
    use std::{fs::File, io::Write as _};

    pub struct Sink(File);

    impl Sink {
        pub fn new(_w: u32, _h: u32) -> Result<Self, Box<dyn std::error::Error>> {
            Ok(Self(File::create("frames.rgba")?))
        }
        pub fn push(&mut self, rgba: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
            self.0.write_all(rgba)?;
            Ok(())
        }
        pub fn close(self) -> Result<String, Box<dyn std::error::Error>> {
            Ok("frames.rgba".to_owned())
        }
    }
}

const W: u32 = 1280;
const H: u32 = 720;
const FRAMES: u32 = 30;

fn main() -> Result<(), Box<dyn Error>> {
    let mut ctx = Ctx::headless(W, H)?;
    ctx.set_clear_color(Color::rgb(18, 18, 22));

    let (Some(cw), Some(ch)) = (NonZeroU32::new(160), NonZeroU32::new(120)) else {
        return Err("bad texture size".into());
    };
    let mut rgba = vec![0u8; 160 * 120 * 4];
    let camera = upload_texture(&ctx, &rgba, cw, ch, Format::Rgba8, Filter::Linear)?;

    let mut orbit = Orbit::new(Meters(1.6));
    let mut ee_path: Vec<WorldPoint> = Vec::with_capacity(FRAMES as usize);
    let mut colored: Vec<(WorldPoint, Color)> = Vec::with_capacity(FRAMES as usize);
    let mut plot: Vec<ImagePoint> = Vec::with_capacity(FRAMES as usize);
    let mut history: Vec<f32> = Vec::with_capacity(FRAMES as usize);

    let mut out = sink::Sink::new(W, H)?;

    for frame in 0..FRAMES {
        let secs = frame as f32 / 30.0;

        // --- the producer, standing in for a decode thread ------------------
        for (i, texel) in rgba.chunks_exact_mut(4).enumerate() {
            let (x, y) = ((i as u32 % 160) as f32, (i as u32 / 160) as f32);
            let wave = ((x * 0.05 + secs * 2.0).sin() * (y * 0.05 - secs).cos() + 1.0) * 0.5;
            texel.copy_from_slice(&[
                (wave * 255.0) as u8,
                (x / 160.0 * 255.0) as u8,
                (y / 120.0 * 255.0) as u8,
                255,
            ]);
        }
        update_texture(&camera, &rgba)?;

        let joints: [f32; 7] = core::array::from_fn(|j| (secs * (1.0 + j as f32 * 0.3)).sin());
        ee_path.push(WorldPoint::new(
            (secs * 0.7).cos() * 0.4,
            (secs * 1.3).sin() * 0.15 + 0.2,
            (secs * 0.7).sin() * 0.4,
        ));
        history.push(joints.first().copied().unwrap_or(0.0));
        orbit.rotate(0.01, 0.0);

        // --- the draw code, byte for byte what the live viewer runs ---------
        let mut f = begin_frame(&mut ctx);
        let [left, right] = f.viewport().split_h();

        let dst = left.inset(8.0).fit_aspect(camera.aspect());
        draw_texture(&mut f, &camera, dst, Color::WHITE);
        draw_bbox(&mut f, dst, Color::rgb(70, 70, 80));

        draw_text_at(
            &mut f,
            ImagePoint::new(left.x + 16.0, left.y + 16.0),
            &format!("frame {frame:>3}   t = {secs:6.2}s"),
            TextScale::X2,
            Color::WHITE,
        );

        let plot_rect = Rect::new(left.x + 16.0, left.bottom() - 96.0, left.w - 32.0, 80.0);
        draw_bbox(&mut f, plot_rect, Color::rgb(50, 50, 60));
        plot.clear();
        plot.extend(history.iter().enumerate().map(|(i, &v)| {
            ImagePoint::new(
                plot_rect.x + plot_rect.w * i as f32 / FRAMES as f32,
                plot_rect.y + plot_rect.h * 0.5 - v * plot_rect.h * 0.45,
            )
        }));
        draw_line_strip_2d(&mut f, &plot, Color::GREEN);

        let cam = orbit.camera(right.w / right.h);
        let mut s = f.scene_in(&cam, right);
        draw_grid(&mut s, 20, Meters(0.1), Color::rgb(45, 45, 55));

        colored.clear();
        let n = ee_path.len().max(1) as f32;
        colored.extend(
            ee_path
                .iter()
                .enumerate()
                .map(|(i, &p)| (p, Color::BLUE.lerp(Color::CYAN, i as f32 / n))),
        );
        draw_line_strip_3d_vc(&mut s, &colored);

        let mut t = Mat4::IDENTITY;
        let links: [Mat4; 7] = core::array::from_fn(|i| {
            let axis = if i % 2 == 0 { Vec3::Y } else { Vec3::X };
            let q = joints.get(i).copied().unwrap_or(0.0);
            t *= Mat4::from_translation(Vec3::new(0.0, 0.12, 0.0))
                * Mat4::from_axis_angle(axis, q * 0.4);
            t
        });
        draw_frames(&mut s, &links, Meters(0.06));

        if let Some(&tip) = ee_path.last() {
            draw_points_3d(&mut s, &[(tip, Color::YELLOW)], Meters(0.02));
        }
        s.end();
        f.end();

        // --- the only other difference: read back instead of present --------
        let pixels = ctx
            .read_pixels()
            .ok_or("headless context should read back")?;
        out.push(&pixels)?;
    }

    let written = out.close()?;
    println!(
        "wrote {FRAMES} frames of {W}x{H} to {written}, {} vertices peak",
        ctx.peak_vertices()
    );
    Ok(())
}
