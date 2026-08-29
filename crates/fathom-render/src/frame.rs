//! Frame and scene typestates.
//!
//! The protocol is encoded in the types, so misuse does not compile and no
//! runtime check is needed: `draw_*` take a `&mut Frame`, so drawing outside a
//! frame is a compile error; 3D primitives take a `&mut Scene`, so drawing in
//! world space without a bound camera is a compile error; and `end` consumes
//! `self`, so there is no double-end and no forgotten end.

use std::sync::Arc;

use fathom_core::Rect;
use glam::{Mat4, Vec2, Vec3, Vec4};

use crate::{
    Camera, Ctx,
    ctx::Surfaced,
    vertex::{Topology, Vertex},
};

/// Begin a frame. Everything drawn before [`Frame::end`] lands in one submit.
///
/// The loop is yours: fathom never calls back into your code.
///
/// ```no_run
/// use fathom_render::{Ctx, begin_frame};
///
/// # fn demo(ctx: &mut Ctx) {
/// let f = begin_frame(ctx);
/// // ... draw ...
/// f.end();
/// # }
/// ```
pub fn begin_frame(ctx: &mut Ctx) -> Frame<'_> {
    ctx.batcher.begin_frame();
    let size = ctx.size();
    let target = ctx.acquire();
    Frame { ctx, target, size }
}

/// An in-progress frame. Drawing needs one; you cannot make one yourself.
#[derive(Debug)]
pub struct Frame<'a> {
    ctx: &'a mut Ctx,
    target: Option<Surfaced>,
    size: (u32, u32),
}

impl<'a> Frame<'a> {
    /// Bind a camera and enter world space.
    ///
    /// 3D primitives take the returned [`Scene`], which is how "draw a
    /// trajectory without saying where the camera is" is made unrepresentable.
    pub fn scene<'f>(&'f mut self, cam: &Camera) -> Scene<'f, 'a> {
        let full = self.viewport();
        self.scene_in(cam, full)
    }

    /// Bind a camera to one panel of the window.
    ///
    /// The 3D half of `Rect` layout: the scene is viewported and scissored to
    /// `panel`, so it sits beside a video stream instead of covering it. Pass
    /// the panel's own aspect ratio to the camera, or the view will be
    /// stretched.
    ///
    /// ```no_run
    /// # use fathom_render::{Ctx, Orbit, begin_frame};
    /// # fn demo(ctx: &mut Ctx, orbit: &Orbit) {
    /// let mut f = begin_frame(ctx);
    /// let [left, right] = f.viewport().split_h();
    /// let mut s = f.scene_in(&orbit.camera(right.w / right.h), right);
    /// // ... 3D lands only in the right half ...
    /// s.end();
    /// f.end();
    /// # }
    /// ```
    pub fn scene_in<'f>(&'f mut self, cam: &Camera, panel: Rect) -> Scene<'f, 'a> {
        Scene {
            view_proj: cam.view_proj(),
            right: cam.right(),
            up: cam.up(),
            viewport: [panel.x, panel.y, panel.w, panel.h],
            frame: self,
        }
    }

    /// The whole drawing area, as a rectangle to carve panels out of.
    ///
    /// Layout is `Rect` math in your code; there is no panel manager.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn viewport(&self) -> Rect {
        Rect::new(0.0, 0.0, self.size.0 as f32, self.size.1 as f32)
    }

    /// Finish the frame: pack, submit once, present.
    ///
    /// Consumes `self`, so the frame cannot be ended twice or drawn into after.
    pub fn end(mut self) {
        if let Some(target) = self.target.take() {
            self.ctx.submit(&target);
            target.present();
        }
    }

    /// Map a pixel position to clip space. 2D sits at the near plane, so a HUD
    /// drawn after a scene lands on top of it.
    #[inline]
    #[allow(clippy::cast_precision_loss)]
    pub(crate) fn clip(&self, p: Vec2) -> Vec4 {
        let (w, h) = (self.size.0 as f32, self.size.1 as f32);
        Vec4::new(2.0 * p.x / w - 1.0, 1.0 - 2.0 * p.y / h, 0.0, 1.0)
    }

    #[inline]
    pub(crate) fn begin(
        &mut self,
        topology: Topology,
        texture: Option<&Arc<wgpu::BindGroup>>,
        count: usize,
    ) {
        #[allow(clippy::cast_precision_loss)]
        let full = [0.0, 0.0, self.size.0 as f32, self.size.1 as f32];
        let (device, queue, batcher) = self.ctx.parts();
        batcher.begin(device, queue, topology, texture, full, count);
    }

    /// Same, but confined to a scene's panel.
    #[inline]
    pub(crate) fn begin_in(&mut self, topology: Topology, viewport: [f32; 4], count: usize) {
        let (device, queue, batcher) = self.ctx.parts();
        batcher.begin(device, queue, topology, None, viewport, count);
    }

    #[inline]
    pub(crate) fn push(&mut self, v: Vertex) {
        self.ctx.batcher.push(v);
    }

    #[inline]
    pub(crate) fn batch_limit(&self) -> usize {
        self.ctx.batcher.batch_limit()
    }
}

/// A frame with a camera bound: the world-space drawing surface.
///
/// Borrows the [`Frame`] mutably, so 2D and 3D cannot interleave within one
/// scene and the draw order stays exactly what you wrote.
#[derive(Debug)]
pub struct Scene<'f, 'a> {
    frame: &'f mut Frame<'a>,
    view_proj: Mat4,
    right: Vec3,
    up: Vec3,
    viewport: [f32; 4],
}

impl Scene<'_, '_> {
    /// Release the camera and return to 2D. Purely for symmetry with
    /// [`Frame::end`]; dropping the scene does the same thing.
    #[allow(clippy::unused_self)] // consuming `self` is the point: it ends the borrow
    pub fn end(self) {}

    /// Map a world position to clip space. The perspective divide is the GPU's.
    #[inline]
    pub(crate) fn clip(&self, p: Vec3) -> Vec4 {
        self.view_proj * p.extend(1.0)
    }

    /// The camera's right and up axes in world space, for billboarding points.
    #[inline]
    pub(crate) const fn basis(&self) -> (Vec3, Vec3) {
        (self.right, self.up)
    }

    #[inline]
    pub(crate) fn begin(&mut self, topology: Topology, count: usize) {
        self.frame.begin_in(topology, self.viewport, count);
    }

    #[inline]
    pub(crate) fn push(&mut self, v: Vertex) {
        self.frame.push(v);
    }

    #[inline]
    pub(crate) fn batch_limit(&self) -> usize {
        self.frame.batch_limit()
    }
}
