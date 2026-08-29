//! `upload_texture` takes a `&Ctx`, which the in-flight `Frame` has borrowed
//! mutably: allocating mid-frame is a compile error, not a stall you profile.
use std::num::NonZeroU32;

use fathom::prelude::*;

fn demo(ctx: &mut Ctx) {
    let f = begin_frame(ctx);
    let (w, h) = (NonZeroU32::MIN, NonZeroU32::MIN);
    let _ = upload_texture(ctx, &[0, 0, 0, 0], w, h, Format::Rgba8, Filter::Linear);
    f.end();
}

fn main() {}
