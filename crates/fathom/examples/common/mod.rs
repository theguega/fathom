//! A minimal window runner shared by the demo examples.
//!
//! This is caller code, not library code: `live_viewer` spells the same loop
//! out in full. It exists here only so the other examples can be what they are
//! meant to be - twenty lines of drawing, with nothing else in the way. If one
//! of them needed more than that, it would signal a missing primitive.
#![allow(dead_code, unreachable_pub)] // each example uses a different part of the harness
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
//  ^ example code: every cast here is small and deliberate

use std::{error::Error, sync::Arc, time::Instant};

use fathom::prelude::*;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

/// What the harness hands your draw function each frame.
pub struct Input {
    /// Turntable camera, driven by left-drag and the wheel.
    pub orbit: Orbit,
    /// Seconds since the window opened.
    pub time: f32,
    /// Cursor position in window pixels.
    pub cursor: ImagePoint,
    /// Whether the left button is currently down.
    pub clicking: bool,
    /// Left-button presses, in order, cleared by the example when it likes.
    pub clicks: Vec<ImagePoint>,
}

struct App<F> {
    title: String,
    window: Option<Arc<Window>>,
    ctx: Option<Ctx>,
    input: Input,
    draw: F,
    start: Instant,
    dragging: bool,
    last: (f64, f64),
}

impl<F: FnMut(&mut Ctx, &mut Input)> ApplicationHandler for App<F> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes().with_title(self.title.clone());
        let Ok(window) = event_loop.create_window(attrs) else {
            event_loop.exit();
            return;
        };
        let window = Arc::new(window);
        let size = window.inner_size();
        match Ctx::new(Arc::clone(&window), size.width, size.height) {
            Ok(mut ctx) => {
                ctx.set_clear_color(Color::rgb(18, 18, 22));
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
                    self.input.clicking = self.dragging;
                    if self.dragging {
                        self.input.clicks.push(self.input.cursor);
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let (dx, dy) = (position.x - self.last.0, position.y - self.last.1);
                self.last = (position.x, position.y);
                self.input.cursor = ImagePoint::new(position.x as f32, position.y as f32);
                if self.dragging {
                    self.input
                        .orbit
                        .rotate(-dx as f32 * 0.01, -dy as f32 * 0.01);
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let notches = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.05,
                };
                self.input.orbit.zoom(-notches);
            }
            WindowEvent::RedrawRequested => {
                self.input.time = self.start.elapsed().as_secs_f32();
                if let Some(ctx) = &mut self.ctx {
                    (self.draw)(ctx, &mut self.input);
                }
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

/// Open a window and call `draw` every frame until it is closed.
///
/// # Errors
///
/// If the event loop cannot be created or the platform refuses a window.
pub fn run<F>(title: &str, distance: Meters, draw: F) -> Result<(), Box<dyn Error>>
where
    F: FnMut(&mut Ctx, &mut Input),
{
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App {
        title: title.to_owned(),
        window: None,
        ctx: None,
        input: Input {
            orbit: Orbit::new(distance),
            time: 0.0,
            cursor: ImagePoint::ORIGIN,
            clicking: false,
            clicks: Vec::with_capacity(8),
        },
        draw,
        start: Instant::now(),
        dragging: false,
        last: (0.0, 0.0),
    })?;
    Ok(())
}
