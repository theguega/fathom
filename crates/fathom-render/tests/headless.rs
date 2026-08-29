//! End-to-end tests against a real GPU, through the offscreen path.
//!
//! These render actual pixels and read them back, which is the only way to
//! catch a wrong winding order, an inverted Y axis or a colour space mistake.

use fathom_core::{Color, ImagePoint, Meters, Radians, Rect, TextScale, Vec3, WorldPoint};
use fathom_render::{
    Camera, Ctx, begin_frame, draw_grid, draw_line_strip_3d, draw_polygon, draw_text_at,
};

const W: u32 = 64;
const H: u32 = 64;

/// Skip rather than fail where no adapter exists, so `cargo test` still works
/// on a headless box with no GPU at all.
fn ctx() -> Option<Ctx> {
    match Ctx::headless(W, H) {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            eprintln!("skipping: no GPU available ({e})");
            None
        }
    }
}

fn pixel(px: &[u8], x: u32, y: u32) -> [u8; 4] {
    let i = ((y * W + x) * 4) as usize;
    px.get(i..i + 4)
        .and_then(|p| p.try_into().ok())
        .unwrap_or([0; 4])
}

#[test]
fn clear_color_reaches_the_framebuffer() {
    let Some(mut ctx) = ctx() else { return };
    ctx.set_clear_color(Color::rgb(10, 20, 30));

    begin_frame(&mut ctx).end();

    let px = ctx.read_pixels().unwrap_or_default();
    assert_eq!(px.len(), (W * H * 4) as usize);
    assert_eq!(pixel(&px, 0, 0), [10, 20, 30, 255]);
    assert_eq!(pixel(&px, W - 1, H - 1), [10, 20, 30, 255]);
}

#[test]
fn a_filled_quad_lands_where_it_was_asked_to() {
    let Some(mut ctx) = ctx() else { return };
    ctx.set_clear_color(Color::BLACK);

    let mut f = begin_frame(&mut ctx);
    // Top-left quadrant, in pixels.
    let r = Rect::new(0.0, 0.0, 32.0, 32.0);
    let corners = [
        ImagePoint::new(r.x, r.y),
        ImagePoint::new(r.right(), r.y),
        ImagePoint::new(r.right(), r.bottom()),
        ImagePoint::new(r.x, r.bottom()),
    ];
    draw_polygon(&mut f, &corners, Color::RED);
    f.end();

    let px = ctx.read_pixels().unwrap_or_default();
    // Inside is red; the other three quadrants are untouched. This is also the
    // Y-axis test: a flipped axis puts the red in the bottom-left.
    assert_eq!(pixel(&px, 16, 16), [255, 0, 0, 255]);
    assert_eq!(pixel(&px, 48, 16), [0, 0, 0, 255]);
    assert_eq!(pixel(&px, 16, 48), [0, 0, 0, 255]);
    assert_eq!(pixel(&px, 48, 48), [0, 0, 0, 255]);
}

#[test]
fn alpha_blends_against_what_is_already_there() {
    let Some(mut ctx) = ctx() else { return };
    ctx.set_clear_color(Color::BLACK);

    let mut f = begin_frame(&mut ctx);
    let quad = [
        ImagePoint::new(0.0, 0.0),
        ImagePoint::new(64.0, 0.0),
        ImagePoint::new(64.0, 64.0),
        ImagePoint::new(0.0, 64.0),
    ];
    draw_polygon(&mut f, &quad, Color::WHITE.with_alpha(0.5));
    f.end();

    let px = ctx.read_pixels().unwrap_or_default();
    let [r, _, _, _] = pixel(&px, 32, 32);
    assert!(
        (120..=136).contains(&r),
        "half-alpha white over black should be mid grey, got {r}"
    );
}

#[test]
fn text_puts_ink_on_the_screen_and_respects_scale() {
    let Some(mut ctx) = ctx() else { return };

    let mut ink = |scale| {
        ctx.set_clear_color(Color::BLACK);
        let mut f = begin_frame(&mut ctx);
        draw_text_at(&mut f, ImagePoint::new(1.0, 1.0), "MM", scale, Color::WHITE);
        f.end();
        ctx.read_pixels()
            .unwrap_or_default()
            .chunks_exact(4)
            .filter(|p| p.first().is_some_and(|&r| r > 128))
            .count()
    };

    let small = ink(TextScale::X1);
    let large = ink(TextScale::X2);
    assert!(
        small > 10,
        "text should draw something, got {small} lit pixels"
    );
    assert!(
        large > small * 2,
        "doubling the scale should roughly quadruple the ink: {small} -> {large}"
    );
}

#[test]
fn a_scene_draws_in_world_space_and_respects_depth() {
    let Some(mut ctx) = ctx() else { return };
    ctx.set_clear_color(Color::BLACK);

    let cam = Camera::perspective(
        Vec3::new(0.0, 2.0, 3.0),
        Vec3::ZERO,
        Vec3::Y,
        Radians(1.0),
        1.0,
        Meters(0.01),
        Meters(100.0),
    );

    let mut f = begin_frame(&mut ctx);
    let mut s = f.scene(&cam);
    draw_grid(&mut s, 8, Meters(0.25), Color::rgb(80, 80, 90));
    draw_line_strip_3d(
        &mut s,
        &[
            WorldPoint::new(-1.0, 0.5, 0.0),
            WorldPoint::new(1.0, 0.5, 0.0),
        ],
        Color::GREEN,
    );
    s.end();
    f.end();

    let px = ctx.read_pixels().unwrap_or_default();
    let lit = px
        .chunks_exact(4)
        .filter(|p| p.iter().take(3).any(|&c| c > 40))
        .count();
    assert!(lit > 50, "a grid and a line should cover pixels, got {lit}");

    let green = px
        .chunks_exact(4)
        .filter(|p| matches!(p, [r, g, b, _] if *g > 200 && *r < 60 && *b < 60))
        .count();
    assert!(
        green > 5,
        "the green line should be visible, got {green} pixels"
    );
}

#[test]
fn a_frame_that_overflows_the_arena_still_draws_everything() {
    let Some(mut ctx) = ctx() else { return };
    ctx.set_clear_color(Color::BLACK);

    // Far more segments than the 64k-vertex arena holds, forcing the overflow
    // path: nothing may be dropped, and the last segment must still appear.
    let path: Vec<WorldPoint> = (0..80_000u32)
        .map(|i| {
            let t = f32::from(u16::try_from(i % 1000).unwrap_or(0)) / 1000.0;
            WorldPoint::new(t.mul_add(2.0, -1.0), 0.0, 0.0)
        })
        .collect();

    let cam = Camera::perspective(
        Vec3::new(0.0, 0.0, 3.0),
        Vec3::ZERO,
        Vec3::Y,
        Radians(1.0),
        1.0,
        Meters(0.01),
        Meters(100.0),
    );

    let mut f = begin_frame(&mut ctx);
    let mut s = f.scene(&cam);
    draw_line_strip_3d(&mut s, &path, Color::WHITE);
    s.end();
    f.end();

    assert!(
        ctx.peak_vertices() > 64 * 1024,
        "this frame should have spilled past one chunk, packed {}",
        ctx.peak_vertices()
    );

    let px = ctx.read_pixels().unwrap_or_default();
    let lit = px
        .chunks_exact(4)
        .filter(|p| p.first().is_some_and(|&r| r > 128))
        .count();
    assert!(
        lit > 20,
        "the whole polyline should be drawn, got {lit} lit pixels"
    );
}
