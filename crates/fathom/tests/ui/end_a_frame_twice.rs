//! `Frame::end` consumes self, so there is no double-end and no use-after-end.
use fathom::prelude::*;

fn demo(ctx: &mut Ctx) {
    let f = begin_frame(ctx);
    f.end();
    f.end();
}

fn main() {}
