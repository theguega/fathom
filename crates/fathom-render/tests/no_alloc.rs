//! The arcade discipline, enforced mechanically rather than by review.
//!
//! The arena is sized at init, so the draw calls between `begin_frame` and
//! `end` must not touch the heap at all. A counting allocator watches the
//! stretch of the frame that fathom controls: acquiring a swapchain image and
//! submitting are wgpu's business and do allocate, so the window is opened
//! after `begin_frame` returns and closed before `end` is called.

use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
};

use fathom_core::{
    Color, ImagePoint, Mat3, Mat4, Meters, Radians, Rect, TextScale, Vec3, WorldPoint,
};
use fathom_render::{
    Camera, Ctx, begin_frame, draw_bbox, draw_frames, draw_grid, draw_line_2d, draw_line_3d,
    draw_line_strip_2d, draw_line_strip_3d, draw_line_strip_3d_vc, draw_points_3d, draw_polygon,
    draw_text_at, draw_wire_ellipsoid,
};

struct Counting;

static WATCHING: AtomicBool = AtomicBool::new(false);
static ALLOCS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every method forwards directly to the system allocator with the same
// arguments; the counter only observes.
#[allow(unsafe_code)]
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if WATCHING.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: forwarding an unmodified layout to the system allocator.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: forwarding a pointer and layout this allocator produced.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if WATCHING.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: forwarding a pointer and layout this allocator produced.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

#[test]
fn drawing_a_frame_never_touches_the_heap() {
    let Ok(mut ctx) = Ctx::headless(320, 240) else {
        eprintln!("skipping: no GPU available");
        return;
    };

    // Caller-owned buffers, filled once. This is what the discipline asks of
    // the caller in return: allocate at init, reuse forever.
    let path: Vec<WorldPoint> = (0..2048u16)
        .map(|i| {
            let t = f32::from(i) / 2048.0;
            WorldPoint::new(t, t * 0.5, t * 0.25)
        })
        .collect();
    let colored: Vec<(WorldPoint, Color)> = path.iter().map(|&p| (p, Color::CYAN)).collect();
    let screen: Vec<ImagePoint> = (0..2048u16)
        .map(|i| ImagePoint::new(f32::from(i) * 0.1, f32::from(i) * 0.05))
        .collect();
    let links: [Mat4; 7] = core::array::from_fn(|i| {
        #[allow(clippy::cast_precision_loss)]
        Mat4::from_translation(Vec3::splat(i as f32 * 0.1))
    });
    let label = "0123456789 abcdefghijklmnopqrstuvwxyz";

    let cam = Camera::perspective(
        Vec3::new(1.0, 1.0, 1.0),
        Vec3::ZERO,
        Vec3::Y,
        Radians(1.0),
        1.33,
        Meters(0.01),
        Meters(100.0),
    );

    // Warm every lazily-initialized path first, so the measured frame is a
    // steady-state one rather than the very first.
    for _ in 0..2 {
        let mut f = begin_frame(&mut ctx);
        draw_text_at(
            &mut f,
            ImagePoint::ORIGIN,
            label,
            TextScale::X1,
            Color::WHITE,
        );
        let mut s = f.scene(&cam);
        draw_line_strip_3d(&mut s, &path, Color::GREEN);
        s.end();
        f.end();
    }

    let mut f = begin_frame(&mut ctx);

    ALLOCS.store(0, Ordering::Relaxed);
    WATCHING.store(true, Ordering::Relaxed);

    draw_line_2d(
        &mut f,
        ImagePoint::ORIGIN,
        ImagePoint::new(10.0, 10.0),
        Color::RED,
    );
    draw_line_strip_2d(&mut f, &screen, Color::GREEN);
    draw_bbox(&mut f, Rect::new(4.0, 4.0, 100.0, 50.0), Color::BLUE);
    draw_polygon(&mut f, &screen[..64], Color::MAGENTA);
    draw_text_at(
        &mut f,
        ImagePoint::new(2.0, 2.0),
        label,
        TextScale::X2,
        Color::WHITE,
    );

    let mut s = f.scene(&cam);
    draw_grid(&mut s, 32, Meters(0.1), Color::GRAY);
    draw_line_3d(
        &mut s,
        WorldPoint::ORIGIN,
        WorldPoint::new(1.0, 1.0, 1.0),
        Color::RED,
    );
    draw_line_strip_3d(&mut s, &path, Color::GREEN);
    draw_line_strip_3d_vc(&mut s, &colored);
    draw_points_3d(&mut s, &colored, Meters(0.01));
    draw_wire_ellipsoid(&mut s, WorldPoint::ORIGIN, Mat3::IDENTITY, Color::YELLOW);
    draw_frames(&mut s, &links, Meters(0.05));
    s.end();

    WATCHING.store(false, Ordering::Relaxed);
    let allocs = ALLOCS.load(Ordering::Relaxed);

    f.end();

    assert_eq!(
        allocs, 0,
        "drawing a frame allocated {allocs} times; the arena is meant to be sized at init"
    );
    assert!(
        ctx.peak_vertices() > 10_000,
        "the test should actually have drawn something"
    );
}
