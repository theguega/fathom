//! Compiles every Rust code block in the book against the real crate.
//!
//! `mdbook test` cannot do this: it has no `--extern`, and its generated
//! doctests default to edition 2015, so `use fathom::prelude::*` does not even
//! resolve. Including the chapters here instead makes them ordinary doctests,
//! run by plain `cargo test --doc`, with the crate and its dependencies
//! resolved by cargo the way they are everywhere else.
//!
//! The crate is `publish = false` and carries no code of its own; the
//! `include_str!` paths reach outside the package, which is exactly why this
//! could not live in `fathom` itself.
#![cfg(doctest)]

macro_rules! chapters {
    ($($name:ident => $path:literal),* $(,)?) => {$(
        #[doc = include_str!($path)]
        mod $name {}
    )*};
}

chapters! {
    introduction    => "../../../book/src/introduction.md",
    philosophy      => "../../../book/src/philosophy.md",
    primitives      => "../../../book/src/primitives.md",
    coordinates     => "../../../book/src/coordinates.md",
    calibration     => "../../../book/src/calibration.md",
    streaming       => "../../../book/src/streaming.md",
    from_matplotlib => "../../../book/src/from-matplotlib.md",
}
