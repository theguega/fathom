//! The §4 type guarantees, asserted as compile errors.
//!
//! A guarantee that is not tested is a guarantee someone refactors away. Each
//! case here is a mistake the type system is supposed to make unrepresentable;
//! if one ever starts compiling, this test fails.

#[test]
fn type_guarantees_hold() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/*.rs");
    #[cfg(feature = "media")]
    t.compile_fail("tests/ui/media/*.rs");
}
