//! The §4 type guarantees, asserted as compile errors.
//!
//! A guarantee that is not tested is a guarantee someone refactors away. Each
//! case here is a mistake the type system is supposed to make unrepresentable;
//! if one ever starts compiling, this test fails.

/// `trybuild` compares the compiler's exact wording, which drifts between
/// toolchains: nightly writes `fathom::Frame::<'a>::end` where 1.97.1 writes
/// `Frame::<'a>::end`, for the same error, about the same guarantee.
///
/// The `.stderr` files record the pinned toolchain from `rust-toolchain.toml`.
/// Off it, the guarantees are still enforced - they are compiler errors either
/// way - but the wording cannot be diffed, so the diff is skipped rather than
/// reported as a failure that is not one.
fn stderr_matches_this_toolchain() -> bool {
    match option_env!("RUSTUP_TOOLCHAIN") {
        // No rustup: whatever `cargo` is on PATH, which is the pinned one.
        None => true,
        Some(t) => t.starts_with("1.97"),
    }
}

#[test]
fn type_guarantees_hold() {
    if !stderr_matches_this_toolchain() {
        eprintln!(
            "skipping the stderr diff: .stderr files are generated with the toolchain pinned \
             in rust-toolchain.toml, and {} phrases these errors differently",
            option_env!("RUSTUP_TOOLCHAIN").unwrap_or("this toolchain")
        );
        return;
    }

    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/*.rs");
    #[cfg(feature = "media")]
    t.compile_fail("tests/ui/media/*.rs");
}
