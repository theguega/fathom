//! `warp` takes a point on a plane. Handing it a world point is the extrinsics
//! mix-up that otherwise costs an afternoon, so it must be a compile error.
use fathom::prelude::*;

fn demo(h: &Homography) {
    let _ = warp(WorldPoint::new(1.0, 2.0, 3.0), h);
}

fn main() {}
