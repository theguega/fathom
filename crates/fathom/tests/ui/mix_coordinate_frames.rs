//! A point carries the frame it lives in; two frames must not mix.
use fathom::prelude::*;

fn takes_world(_: WorldPoint) {}

fn demo() {
    takes_world(CameraPoint::new(1.0, 2.0, 3.0));
    let _: ImagePoint = PlanePoint::new(1.0, 2.0);
}

fn main() {}
