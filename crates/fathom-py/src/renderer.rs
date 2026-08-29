//! The handles Python holds: a renderer, a texture, a camera.

use std::{num::NonZeroU32, sync::Arc};

use fathom_core::{Meters, Radians, Rect, Vec3};
use fathom_render as fr;
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    platform::pump_events::EventLoopExtPumpEvents as _,
    window::{Window, WindowId},
};

use crate::convert;

/// A GPU texture. Cloning is cheap; clones address the same allocation.
///
/// `unsendable` because a texture belongs to the [`Renderer`] that made it, and
/// that is unsendable too. It also sidesteps asking whether the whole wgpu
/// handle graph is `Send`, a query that recurses far enough to hit the
/// compiler's limit.
#[pyclass(module = "fathom", from_py_object, unsendable)]
#[derive(Clone, Debug)]
pub struct Texture(pub(crate) fr::Texture);

#[pymethods]
impl Texture {
    /// Width in pixels.
    #[getter]
    const fn width(&self) -> u32 {
        self.0.width()
    }

    /// Height in pixels.
    #[getter]
    const fn height(&self) -> u32 {
        self.0.height()
    }

    /// Width divided by height.
    #[getter]
    fn aspect(&self) -> f32 {
        self.0.aspect()
    }

    /// Replace this texture's pixels from an RGBA8 buffer.
    ///
    /// The whole live-streaming integration surface: a non-blocking staging
    /// write. If no new frame arrived, do not call it and the previous texture
    /// is redrawn.
    fn update(&self, data: &[u8]) -> PyResult<()> {
        fr::update_texture(&self.0, data).map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

/// A bound viewpoint: view and projection, already multiplied.
#[pyclass(module = "fathom", from_py_object)]
#[derive(Clone, Copy, Debug)]
pub struct Camera(pub(crate) fr::Camera);

#[pymethods]
impl Camera {
    /// A perspective camera looking from `eye` at `target`.
    #[new]
    #[pyo3(signature = (eye, target, up=(0.0, 1.0, 0.0), fov_y=0.785, aspect=1.777, near=0.01, far=100.0))]
    fn new(
        eye: (f32, f32, f32),
        target: (f32, f32, f32),
        up: (f32, f32, f32),
        fov_y: f32,
        aspect: f32,
        near: f32,
        far: f32,
    ) -> Self {
        Self(fr::Camera::perspective(
            Vec3::from(eye),
            Vec3::from(target),
            Vec3::from(up),
            Radians(fov_y),
            aspect,
            Meters(near),
            Meters(far),
        ))
    }
}

/// A turntable viewpoint: plain state you drive from your own input handling.
#[pyclass(module = "fathom", from_py_object)]
#[derive(Clone, Copy, Debug)]
pub struct Orbit(pub(crate) fr::Orbit);

#[pymethods]
impl Orbit {
    /// A turntable at `distance` metres from the origin.
    #[new]
    #[pyo3(signature = (distance=2.0))]
    fn new(distance: f32) -> Self {
        Self(fr::Orbit::new(Meters(distance)))
    }

    /// Yaw and pitch by radians; pitch is clamped short of the poles.
    fn rotate(&mut self, dyaw: f32, dpitch: f32) {
        self.0.rotate(dyaw, dpitch);
    }

    /// Scale the distance, so each notch is a constant proportion.
    fn zoom(&mut self, notches: f32) {
        self.0.zoom(notches);
    }

    /// Slide the target across the view plane.
    fn pan(&mut self, dx: f32, dy: f32) {
        self.0.pan(dx, dy);
    }

    /// The camera for this turntable at the given aspect ratio.
    fn camera(&self, aspect: f32) -> Camera {
        Camera(self.0.camera(aspect))
    }

    /// Distance from the target, in metres.
    #[getter]
    fn distance(&self) -> f32 {
        self.0.distance.get()
    }

    /// Set the point the camera looks at and rotates around.
    fn set_target(&mut self, target: (f32, f32, f32)) {
        self.0.target = Vec3::from(target);
    }

    /// Set the world up axis: `(0, 1, 0)` for graphics data, `(0, 0, 1)` for
    /// most robot frames.
    fn set_up(&mut self, up: (f32, f32, f32)) {
        self.0.up = Vec3::from(up);
    }
}

/// The window and its event loop. There is exactly one per process, which is
/// why it is created once in `window()` and never again.
struct Pump {
    event_loop: EventLoop<()>,
    window: Arc<Window>,
}

/// What one round of pumping observed.
#[derive(Default)]
struct Events {
    closed: bool,
    resized: Option<(u32, u32)>,
}

/// Forwards winit's callbacks into plain state, so Python keeps the loop.
struct Handler<'a>(&'a mut Events);

impl ApplicationHandler for Handler<'_> {
    fn resumed(&mut self, _: &ActiveEventLoop) {}

    fn window_event(&mut self, _: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => self.0.closed = true,
            WindowEvent::Resized(size) => self.0.resized = Some((size.width, size.height)),
            _ => {}
        }
    }
}

/// The renderer, and the frame currently in flight.
///
/// Python cannot express the borrow that ties a `Frame` to its `Ctx`, so this
/// holds both and enforces the relationship at runtime instead: every drawing
/// entry point goes through [`Renderer::frame`], which fails cleanly if no
/// frame is open.
#[pyclass(module = "fathom", unsendable)]
#[allow(missing_debug_implementations)] // holds a wgpu context and a live event loop
pub struct Renderer {
    // Declaration order is the invariant: `frame` borrows from `ctx`, and Rust
    // drops fields in declaration order, so the borrow is always released
    // first.
    frame: Option<fr::Frame<'static>>,
    ctx: Box<fr::Ctx>,
    scene: Option<(fr::Camera, Rect)>,
    pump: Option<Pump>,
    closed: bool,
}

#[pymethods]
impl Renderer {
    /// Open a window and a renderer for it.
    ///
    /// You still own the loop: call [`Renderer::poll`] once per iteration and
    /// stop when it returns `False`.
    #[staticmethod]
    #[pyo3(signature = (title="fathom", width=1280, height=720))]
    fn window(title: &str, width: u32, height: u32) -> PyResult<Self> {
        let event_loop = EventLoop::new().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        event_loop.set_control_flow(ControlFlow::Poll);

        #[allow(deprecated)] // the non-deprecated path needs winit to own the loop
        let window = event_loop
            .create_window(
                Window::default_attributes()
                    .with_title(title)
                    .with_inner_size(winit::dpi::LogicalSize::new(width, height)),
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let window = Arc::new(window);
        let size = window.inner_size();

        let ctx = fr::Ctx::new(Arc::clone(&window), size.width, size.height)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(Self {
            frame: None,
            ctx: Box::new(ctx),
            scene: None,
            pump: Some(Pump { event_loop, window }),
            closed: false,
        })
    }

    /// Open a renderer with no window, drawing into an offscreen texture.
    ///
    /// The same draw calls, with no display server. Read the result back with
    /// [`Renderer::read_pixels`].
    #[staticmethod]
    #[pyo3(signature = (width=1280, height=720))]
    fn headless(width: u32, height: u32) -> PyResult<Self> {
        let ctx =
            fr::Ctx::headless(width, height).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            frame: None,
            ctx: Box::new(ctx),
            scene: None,
            pump: None,
            closed: false,
        })
    }

    /// Pump window events. Returns `False` once the window has been closed.
    ///
    /// Always `True` for a headless renderer, which has nothing to close.
    fn poll(&mut self) -> bool {
        let Some(pump) = &mut self.pump else {
            return true;
        };
        let mut events = Events::default();
        pump.event_loop
            .pump_app_events(Some(core::time::Duration::ZERO), &mut Handler(&mut events));
        pump.window.request_redraw();

        // `pump` borrows `self.pump` while these touch `self.ctx` and
        // `self.closed`; naming the fields directly is what lets the borrow
        // checker see they are disjoint.
        if let Some((w, h)) = events.resized {
            self.ctx.resize(w, h);
        }
        self.closed |= events.closed;
        !self.closed
    }

    /// Set the colour the next frame is cleared to.
    #[pyo3(signature = (rgba))]
    fn set_clear_color(&mut self, rgba: (u8, u8, u8, u8)) {
        self.ctx.set_clear_color(convert::color(rgba));
    }

    /// Current drawing size in pixels, as `(width, height)`.
    #[getter]
    fn size(&self) -> (u32, u32) {
        self.ctx.size()
    }

    /// Width divided by height.
    #[getter]
    fn aspect(&self) -> f32 {
        self.ctx.aspect()
    }

    /// Vertices packed by the most recent frame.
    #[getter]
    fn peak_vertices(&self) -> usize {
        self.ctx.peak_vertices()
    }

    /// The whole drawing area as `(x, y, w, h)`, to carve panels out of.
    #[getter]
    fn viewport(&self) -> (f32, f32, f32, f32) {
        let (w, h) = self.ctx.size();
        #[allow(clippy::cast_precision_loss)]
        (0.0, 0.0, w as f32, h as f32)
    }

    /// Upload pixel data to a new GPU texture.
    ///
    /// Allocation happens between frames by construction: this borrows the
    /// renderer, and an open frame has already borrowed it.
    #[pyo3(signature = (data, width, height, nearest=false))]
    fn upload_texture(
        &self,
        data: &[u8],
        width: u32,
        height: u32,
        nearest: bool,
    ) -> PyResult<Texture> {
        let (Some(w), Some(h)) = (NonZeroU32::new(width), NonZeroU32::new(height)) else {
            return Err(PyValueError::new_err("texture dimensions must be non-zero"));
        };
        let filter = if nearest {
            fr::Filter::Nearest
        } else {
            fr::Filter::Linear
        };
        fr::upload_texture(&self.ctx, data, w, h, fr::Format::Rgba8, filter)
            .map(Texture)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Begin a frame. Everything drawn before [`Renderer::end_frame`] lands in
    /// one submit.
    fn begin_frame(&mut self) -> PyResult<()> {
        if self.frame.is_some() {
            return Err(PyRuntimeError::new_err(
                "a frame is already open; call end_frame() first",
            ));
        }
        let frame = fr::begin_frame(&mut self.ctx);
        // SAFETY: `frame` borrows `*self.ctx`, which is boxed and so has a
        // stable address that outlives the borrow. The lifetime is erased only
        // because Python cannot carry it. The invariant that keeps this sound:
        // `frame` is declared before `ctx`, so it is dropped first; `ctx` is
        // never moved, reborrowed mutably, or dropped while `frame` is `Some`,
        // which is enforced by every method going through `self.frame`.
        let frame: fr::Frame<'static> = unsafe { core::mem::transmute(frame) };
        self.frame = Some(frame);
        Ok(())
    }

    /// Bind a camera, so the 3D primitives have a viewpoint.
    ///
    /// Pass `panel` to confine the view to one rectangle of the window; the
    /// camera's aspect ratio should match it.
    #[pyo3(signature = (camera, panel=None))]
    fn scene(&mut self, camera: &Camera, panel: Option<(f32, f32, f32, f32)>) {
        let (w, h) = self.ctx.size();
        #[allow(clippy::cast_precision_loss)]
        let full = Rect::new(0.0, 0.0, w as f32, h as f32);
        self.scene = Some((camera.0, panel.map_or(full, convert::rect)));
    }

    /// Release the bound camera and return to 2D.
    fn end_scene(&mut self) {
        self.scene = None;
    }

    /// Finish the frame: pack, submit once, present.
    fn end_frame(&mut self) -> PyResult<()> {
        self.scene = None;
        self.frame
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("no frame is open"))?
            .end();
        Ok(())
    }

    /// Read the offscreen target back as RGBA8 rows, top row first.
    ///
    /// Returns `None` for a windowed renderer.
    fn read_pixels(&self) -> Option<Vec<u8>> {
        self.ctx.read_pixels()
    }
}

impl Renderer {
    /// The open frame, or a clean Python error.
    pub(crate) fn frame(&mut self) -> PyResult<&mut fr::Frame<'static>> {
        self.frame
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("no frame is open; call begin_frame() first"))
    }

    /// The open frame plus its bound camera, or a clean Python error.
    ///
    /// A `Scene` is rebuilt per call rather than stored: it is two matrices and
    /// a rectangle, so it costs nothing, and it means only one borrow has to be
    /// held across the Python boundary instead of two.
    pub(crate) fn bound_scene(&mut self) -> PyResult<fr::Scene<'_, 'static>> {
        // Frame first: with no frame open, "no frame is open" is the actionable
        // message, and "no camera is bound" would send the caller the wrong way.
        let frame = self
            .frame
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("no frame is open; call begin_frame() first"))?;
        let (camera, panel) = self.scene.ok_or_else(|| {
            PyRuntimeError::new_err("no camera is bound; call scene(camera) first")
        })?;
        Ok(frame.scene_in(&camera, panel))
    }
}
