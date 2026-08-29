//! The one place `unsafe` lives in this crate.
//!
//! Everything else that needs to reach the GPU gets `Pod` from
//! `#[derive(bytemuck::Pod)]`, which checks for padding on your behalf. The
//! derive cannot express a *generic* `Point<F>`, though - it would demand
//! `F: Pod` for a tag that has no bytes at all - so that one impl is written
//! out here, with a `const` size assertion standing in for the padding check
//! the derive would have done.

/// Declare a type plain-old-data, pinning its size so a later field cannot
/// silently introduce padding and quietly invalidate the impl.
macro_rules! pod {
    ($t:ty, $size:expr) => {
        const _: () = assert!(
            ::core::mem::size_of::<$t>() == $size,
            "size changed: re-check that this type is still padding-free before widening the impl",
        );

        // SAFETY: the type is `repr(transparent)` or `repr(C)` over fields that are
        // themselves `Zeroable`, so the all-zero bit pattern is a valid value.
        #[allow(unsafe_code)]
        unsafe impl ::bytemuck::Zeroable for $t {}

        // SAFETY: the type is `repr(transparent)` or `repr(C)` over `Pod` fields, has no
        // padding - pinned by the size assertion above - no invalid bit patterns, and no
        // interior mutability. Zero-sized markers add none of those.
        #[allow(unsafe_code)]
        unsafe impl ::bytemuck::Pod for $t {}
    };
}

pub(crate) use pod;
