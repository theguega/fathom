//! The reference example: you own the loop, you own the producer thread.
//!
//! A worker generates timestamped camera frames and joint states and pushes
//! them down a bounded channel. The draw loop drains what has arrived and
//! draws; if nothing arrived, it does not call `update_texture` and the
//! previous frame is redrawn. A slow producer degrades to a stale image, never
//! to a stalled render loop, because fathom owns no threads and no buffering.
//!
//! Live and playback are the same code path here, because the library has no
//! concept of a "source".
//!
//! Run with `cargo run -p fathom --example live_viewer`.
#![allow(clippy::cast_precision_loss)] // example code: every cast here is small and deliberate

use std::{
    error::Error,
    num::NonZeroU32,
    sync::{Arc, mpsc},
    time::{Duration, Instant},
};

use fathom::prelude::*;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

const CAM_W: u32 = 160;
const CAM_H: u32 = 120;
const HISTORY: usize = 512;

/// What the producer sends. In a real integration this is whatever your MCAP or
/// LeRobot adapter hands back: see the `adapter_stub` example.
struct Sample {
    t: Timestamp,
    rgba: Vec<u8>,
    joints: [f32; 7],
    ee: WorldPoint,
}

/// The example owns this thread. fathom never starts one.
fn spawn_producer() -> mpsc::Receiver<Sample> {
    let (tx, rx) = mpsc::sync_channel(4);
    std::thread::spawn(move || {
        let start = Instant::now();
        let mut rgba = vec![0u8; (CAM_W * CAM_H * 4) as usize];
        loop {
            let secs = start.elapsed().as_secs_f32();
            for (i, texel) in rgba.chunks_exact_mut(4).enumerate() {
                #[allow(clippy::cast_possible_truncation)]
                let (x, y) = ((i as u32 % CAM_W) as f32, (i as u32 / CAM_W) as f32);
                let wave = ((x * 0.05 + secs * 2.0).sin() * (y * 0.05 - secs).cos() + 1.0) * 0.5;
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                texel.copy_from_slice(&[
                    (wave * 255.0) as u8,
                    (x / CAM_W as f32 * 255.0) as u8,
                    (y / CAM_H as f32 * 255.0) as u8,
                    255,
                ]);
            }

            let sample = Sample {
                t: Timestamp::from_secs_f64(f64::from(secs)),
                rgba: rgba.clone(),
                joints: core::array::from_fn(|j| (secs * (1.0 + j as f32 * 0.3)).sin()),
                ee: WorldPoint::new(
                    (secs * 0.7).cos() * 0.4,
                    (secs * 1.3).sin() * 0.15 + 0.2,
                    (secs * 0.7).sin() * 0.4,
                ),
            };
            if tx.send(sample).is_err() {
                return; // viewer closed
            }
            std::thread::sleep(Duration::from_millis(33));
        }
    });
    rx
}

/// Everything the viewer keeps. The library keeps none of it.
struct Viewer {
    window: Option<Arc<Window>>,
    ctx: Option<Ctx>,
    camera: Option<Texture>,
    rx: mpsc::Receiver<Sample>,
    orbit: Orbit,

    // Caller-owned history. Want a trajectory to persist across frames? Keep
    // the Vec and call the draw function every frame.
    ee_path: Vec<WorldPoint>,
    joint0: Vec<f32>,
    now: Timestamp,
    joints: [f32; 7],
    frames: u32,
    fps: f32,
    peak: usize,
    last_fps: Instant,

    // Reusable scratch buffers: allocate once, refill every frame.
    plot: Vec<ImagePoint>,
    colored: Vec<(WorldPoint, Color)>,

    dragging: bool,
    cursor: (f64, f64),
}

impl Viewer {
    fn new() -> Self {
        Self {
            window: None,
            ctx: None,
            camera: None,
            rx: spawn_producer(),
            orbit: Orbit::new(Meters(1.6)),
            ee_path: Vec::with_capacity(HISTORY),
            joint0: Vec::with_capacity(HISTORY),
            now: Timestamp(0),
            joints: [0.0; 7],
            frames: 0,
            fps: 0.0,
            peak: 0,
            last_fps: Instant::now(),
            plot: Vec::with_capacity(HISTORY),
            colored: Vec::with_capacity(HISTORY),
            dragging: false,
            cursor: (0.0, 0.0),
        }
    }

    /// Drain whatever the producer managed to send since the last frame.
    fn pull(&mut self) {
        while let Ok(s) = self.rx.try_recv() {
            self.now = s.t;
            self.joints = s.joints;
            if let Some(tex) = &self.camera {
                // The entire live-streaming integration surface.
                let _ = update_texture(tex, &s.rgba);
            }
            push_capped(&mut self.ee_path, s.ee, HISTORY);
            push_capped(&mut self.joint0, s.joints[0], HISTORY);
        }
    }

    fn draw(&mut self) {
        let Some(ctx) = &mut self.ctx else { return };

        self.frames += 1;
        if self.last_fps.elapsed() >= Duration::from_secs(1) {
            {
                self.fps = self.frames as f32 / self.last_fps.elapsed().as_secs_f32();
            }
            self.frames = 0;
            self.last_fps = Instant::now();
        }

        let mut f = begin_frame(ctx);
        let [left, right] = f.viewport().split_h();

        // --- 2D panel: the camera stream, a HUD, a time series ---------------
        if let Some(tex) = &self.camera {
            let dst = left.inset(8.0).fit_aspect(tex.aspect());
            draw_texture(&mut f, tex, dst, Color::WHITE);
            draw_bbox(&mut f, dst, Color::rgb(70, 70, 80));
        }

        draw_text_at(
            &mut f,
            ImagePoint::new(left.x + 16.0, left.y + 16.0),
            &format!("{:5.1} fps   t = {:7.2}s", self.fps, self.now.as_secs_f64()),
            TextScale::X2,
            Color::WHITE,
        );
        draw_text_at(
            &mut f,
            ImagePoint::new(left.x + 16.0, left.y + 36.0),
            &format!("{} vertices last frame", self.peak),
            TextScale::X1,
            Color::GRAY,
        );

        // A time series is `draw_line_strip_2d` per channel, with your own
        // axis scaling. There is no plotting subsystem.
        let plot = Rect::new(left.x + 16.0, left.bottom() - 96.0, left.w - 32.0, 80.0);
        draw_bbox(&mut f, plot, Color::rgb(50, 50, 60));
        self.plot.clear();
        self.plot
            .extend(self.joint0.iter().enumerate().map(|(i, &v)| {
                ImagePoint::new(
                    plot.x + plot.w * i as f32 / HISTORY as f32,
                    plot.y + plot.h * 0.5 - v * plot.h * 0.45,
                )
            }));
        draw_line_strip_2d(&mut f, &self.plot, Color::GREEN);

        let mut y = plot.y - 16.0 - 8.0 * 7.0;
        for (j, q) in self.joints.iter().enumerate() {
            draw_text_at(
                &mut f,
                ImagePoint::new(left.x + 16.0, y),
                &format!("q{j} {q:+.3}"),
                TextScale::X1,
                Color::rgb(150, 150, 160),
            );
            y += 10.0;
        }

        // --- 3D panel: the same instant, in world space ----------------------
        let cam = self.orbit.camera(right.w / right.h);
        let mut s = f.scene_in(&cam, right);

        draw_grid(&mut s, 20, Meters(0.1), Color::rgb(45, 45, 55));

        // Per-vertex colour is the whole "age gradient" feature, in caller code.
        self.colored.clear();
        let n = self.ee_path.len().max(1) as f32;
        self.colored.extend(
            self.ee_path
                .iter()
                .enumerate()
                .map(|(i, &p)| (p, Color::BLUE.lerp(Color::CYAN, i as f32 / n))),
        );
        draw_line_strip_3d_vc(&mut s, &self.colored);

        // Link transforms come from your control stack; fathom does no FK.
        let links = fake_chain(&self.joints);
        draw_frames(&mut s, &links, Meters(0.06));

        if let Some(&tip) = self.ee_path.last() {
            draw_points_3d(&mut s, &[(tip, Color::YELLOW)], Meters(0.02));
        }
        s.end();

        f.end();

        // Readable once the frame has given the context back: the number the
        // fixed vertex budget is stated against.
        self.peak = ctx.peak_vertices();
    }
}

/// A serial chain, accumulated in the caller's code. This is why no kinematics
/// crate is needed: FK is an accumulate-and-collect with glam already on hand.
fn fake_chain(joints: &[f32; 7]) -> [Mat4; 7] {
    let mut t = Mat4::IDENTITY;
    core::array::from_fn(|i| {
        let axis = if i % 2 == 0 { Vec3::Y } else { Vec3::X };
        let q = joints.get(i).copied().unwrap_or(0.0);
        t *= Mat4::from_translation(Vec3::new(0.0, 0.12, 0.0))
            * Mat4::from_axis_angle(axis, q * 0.4);
        t
    })
}

fn push_capped<T>(buf: &mut Vec<T>, v: T, cap: usize) {
    if buf.len() == cap {
        buf.remove(0);
    }
    buf.push(v);
}

impl ApplicationHandler for Viewer {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes().with_title("fathom - live viewer");
        let Ok(window) = event_loop.create_window(attrs) else {
            event_loop.exit();
            return;
        };
        let window = Arc::new(window);
        let size = window.inner_size();

        match Ctx::new(Arc::clone(&window), size.width, size.height) {
            Ok(mut ctx) => {
                ctx.set_clear_color(Color::rgb(18, 18, 22));
                let (Some(w), Some(h)) = (NonZeroU32::new(CAM_W), NonZeroU32::new(CAM_H)) else {
                    return;
                };
                let blank = vec![0u8; (CAM_W * CAM_H * 4) as usize];
                self.camera =
                    upload_texture(&ctx, &blank, w, h, Format::Rgba8, Filter::Linear).ok();
                self.ctx = Some(ctx);
                self.window = Some(window);
            }
            Err(e) => {
                eprintln!("could not start the renderer: {e}");
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(ctx) = &mut self.ctx {
                    ctx.resize(size.width, size.height);
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Left {
                    self.dragging = state == ElementState::Pressed;
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let (dx, dy) = (position.x - self.cursor.0, position.y - self.cursor.1);
                self.cursor = (position.x, position.y);
                if self.dragging {
                    #[allow(clippy::cast_possible_truncation)]
                    self.orbit.rotate(-dx as f32 * 0.01, -dy as f32 * 0.01);
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let notches = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    #[allow(clippy::cast_possible_truncation)]
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.05,
                };
                self.orbit.zoom(-notches);
            }
            WindowEvent::RedrawRequested => {
                self.pull();
                self.draw();
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut Viewer::new())?;
    Ok(())
}
