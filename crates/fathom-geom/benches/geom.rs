//! The pure-maths hot paths, benchmarked without opening a window.
#![allow(clippy::unwrap_used, clippy::cast_precision_loss)] // benches are an explicit escape hatch
#![allow(missing_docs)] // criterion_group generates an undocumented item

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use fathom_geom::{Color, ColorMap, Intrinsics, Vec3, WorldPoint, colormap_into, look_at, project};

fn colormap(c: &mut Criterion) {
    let values: Vec<f32> = (0..1_000_000).map(|i| (i % 1000) as f32 / 1000.0).collect();
    let mut out = vec![Color::TRANSPARENT; values.len()];

    let mut group = c.benchmark_group("colormap_into/1M");
    for map in [ColorMap::Turbo, ColorMap::Viridis, ColorMap::Grey] {
        group.bench_function(format!("{map:?}"), |b| {
            b.iter(|| colormap_into(black_box(&values), 0.0..1.0, map, black_box(&mut out)));
        });
    }
    group.finish();
}

fn projection(c: &mut Criterion) {
    let plain = Intrinsics::new(600.0, 600.0, 320.0, 240.0).unwrap();
    let wide = plain
        .with_brown_conrady([-0.28, 0.07, 0.001], [0.0005, -0.0002])
        .unwrap();
    let e = look_at(Vec3::new(0.5, 0.5, -2.0), Vec3::ZERO, Vec3::Y).unwrap();

    let points: Vec<WorldPoint> = (0..100_000)
        .map(|i| {
            let t = (i % 1000) as f32 / 1000.0;
            WorldPoint::new(t - 0.5, t * 0.3, t + 0.1)
        })
        .collect();

    let mut group = c.benchmark_group("project/100k");
    group.bench_function("undistorted", |b| {
        b.iter(|| {
            for p in black_box(&points) {
                black_box(project(*p, &plain, &e));
            }
        });
    });
    group.bench_function("brown_conrady", |b| {
        b.iter(|| {
            for p in black_box(&points) {
                black_box(project(*p, &wide, &e));
            }
        });
    });
    group.finish();
}

criterion_group!(benches, colormap, projection);
criterion_main!(benches);
