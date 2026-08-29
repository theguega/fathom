//! 3D primitives need a bound camera. Passing a `Frame` must not compile.
use fathom::prelude::*;

fn demo(ctx: &mut Ctx) {
    let mut f = begin_frame(ctx);
    draw_line_3d(&mut f, WorldPoint::ORIGIN, WorldPoint::ORIGIN, Color::RED);
    f.end();
}

fn main() {}
