//! Vertex packing against the stated budget: under 1ms of CPU for 100k
//! vertices, one submit per frame.
//!
//! This measures the draw calls only. Acquiring a target and submitting are
//! wgpu's cost, not the packing loop's, so the timed region is exactly the
//! stretch between `begin_frame` and `end`.
#![allow(clippy::unwrap_used, clippy::cast_precision_loss)] // benches are an explicit escape hatch
#![allow(missing_docs)] // criterion_group generates an undocumented item

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use fathom_core::{Color, ImagePoint, Meters, Radians, TextScale, Vec3, WorldPoint};
use fathom_render::{Camera, Ctx, begin_frame, draw_line_strip_3d, draw_points_3d, draw_text_at};

fn packing(c: &mut Criterion) {
    let Ok(mut ctx) = Ctx::headless(1280, 720) else {
        eprintln!("skipping benches: no GPU available");
        return;
    };

    let cam = Camera::perspective(
        Vec3::new(2.0, 2.0, 2.0),
        Vec3::ZERO,
        Vec3::Y,
        Radians(1.0),
        16.0 / 9.0,
        Meters(0.01),
        Meters(100.0),
    );

    // 50k points is 100k vertices through the line pipeline: the budget figure.
    let path: Vec<WorldPoint> = (0..50_000)
        .map(|i| {
            let t = (i % 997) as f32 / 997.0;
            let a = t * core::f32::consts::TAU;
            WorldPoint::new(t - 0.5, a.sin() * 0.3, a.cos() * 0.3)
        })
        .collect();

    // A cloud of the same vertex count: 16k points at six vertices each.
    let cloud: Vec<(WorldPoint, Color)> = path
        .iter()
        .take(16_666)
        .map(|&p| (p, Color::CYAN))
        .collect();

    let mut group = c.benchmark_group("pack");
    group.bench_function("line_strip_3d/100k_vertices", |b| {
        b.iter(|| {
            let mut f = begin_frame(&mut ctx);
            let mut s = f.scene(&cam);
            draw_line_strip_3d(&mut s, black_box(&path), Color::GREEN);
            s.end();
            f.end();
        });
    });
    group.bench_function("points_3d/100k_vertices", |b| {
        b.iter(|| {
            let mut f = begin_frame(&mut ctx);
            let mut s = f.scene(&cam);
            draw_points_3d(&mut s, black_box(&cloud), Meters(0.005));
            s.end();
            f.end();
        });
    });
    group.bench_function("text/1k_glyphs", |b| {
        let line = "the quick brown fox jumps over the lazy dog 0123456789 ".repeat(19);
        b.iter(|| {
            let mut f = begin_frame(&mut ctx);
            draw_text_at(
                &mut f,
                ImagePoint::ORIGIN,
                black_box(&line),
                TextScale::X1,
                Color::WHITE,
            );
            f.end();
        });
    });
    group.finish();
}

criterion_group!(benches, packing);
criterion_main!(benches);
