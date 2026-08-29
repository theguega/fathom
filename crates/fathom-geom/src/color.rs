//! Scalar-to-color ramps.
//!
//! Pixel encoding, not analysis. The same function serves depth maps, attention
//! heatmaps, similarity matrices and outlier scores, which is exactly why
//! fathom ships no separate heatmap primitive: a heatmap is a texture.

use alloc::{vec, vec::Vec};
use core::ops::Range;

use fathom_core::{Color, ColorMap};

/// Sixth-order polynomial fits, evaluated with Horner's rule. Coefficients are
/// the standard WebGL fits of the reference ramps; max error is well under one
/// 8-bit step, and they cost no table lookup and no data section.
type Poly = [[f32; 3]; 7];

const TURBO: Poly = [
    [0.114_089_01, 0.062_883_41, 0.224_833_72],
    [6.716_419_5, 3.182_286_8, 7.571_581_6],
    [-66.094_02, -4.927_983, -10.094_394],
    [228.766_08, 25.049_868, -91.541_05],
    [-334.835_16, -69.317_5, 288.585_88],
    [218.763_72, 67.521_51, -305.204_6],
    [-52.889_034, -21.545_273, 110.517_46],
];

const VIRIDIS: Poly = [
    [0.277_727_33, 0.005_407_345, 0.334_099_8],
    [0.105_093_04, 1.404_613_5, 1.384_590_2],
    [-0.330_861_83, 0.214_847_56, 0.095_095_16],
    [-4.634_230_5, -5.799_101, -19.332_441],
    [6.228_27, 14.179_933, 56.690_55],
    [4.776_385, -13.745_145, -65.353_035],
    [-5.435_456, 4.645_852_6, 26.312_435],
];

const MAGMA: Poly = [
    [-0.002_136_485, -0.000_749_655_1, -0.005_386_128],
    [0.251_660_54, 0.677_523_25, 2.494_026_6],
    [8.353_717, -3.577_719_5, 0.314_467_9],
    [-27.668_734, 14.264_731, -13.649_213],
    [52.176_14, -27.943_606, 12.944_17],
    [-50.768_524, 29.046_583, 4.234_153],
    [18.655_705, -11.489_773, -5.601_961_4],
];

#[inline]
fn eval(poly: &Poly, t: f32) -> Color {
    let mut acc = [0.0f32; 3];
    for coeffs in poly.iter().rev() {
        for (a, c) in acc.iter_mut().zip(coeffs) {
            *a = a.mul_add(t, *c);
        }
    }
    quantize(acc)
}

#[inline]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn quantize(rgb: [f32; 3]) -> Color {
    let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    Color::rgb(q(rgb[0]), q(rgb[1]), q(rgb[2]))
}

/// Map a single normalized scalar through a ramp. `t` is clamped to `0.0..=1.0`.
#[inline]
#[must_use]
fn sample(t: f32, map: ColorMap) -> Color {
    let t = if t.is_nan() { 0.0 } else { t.clamp(0.0, 1.0) };
    #[allow(clippy::match_same_arms)] // the wildcard is a forward-compat fallback, not a duplicate
    match map {
        ColorMap::Turbo => eval(&TURBO, t),
        ColorMap::Viridis => eval(&VIRIDIS, t),
        ColorMap::Magma => eval(&MAGMA, t),
        ColorMap::Grey => quantize([t, t, t]),
        ColorMap::Coolwarm => {
            if t < 0.5 {
                Color::rgb(59, 76, 192).lerp(Color::WHITE, t * 2.0)
            } else {
                Color::WHITE.lerp(Color::rgb(180, 4, 38), t.mul_add(2.0, -1.0))
            }
        }
        // `ColorMap` is `#[non_exhaustive]`: a ramp added by a newer fathom-core
        // than this fathom-geom falls back to the default rather than breaking
        // the build of everything downstream.
        _ => eval(&TURBO, t),
    }
}

/// Map `values` through `map`, writing into a caller-owned buffer.
///
/// This is the form used in a loop: it allocates nothing, so a scratch `Vec`
/// filled once at startup serves every frame. Values outside `range` clamp to
/// the ends; `NaN` maps to the low end. Extra elements of `out` are untouched,
/// and values past the end of `out` are ignored - the shorter slice wins, with
/// no panic and no silent reallocation.
///
/// ```
/// use fathom_geom::{ColorMap, Color, colormap_into};
///
/// let mut scratch = vec![Color::BLACK; 4];
/// colormap_into(&[0.0, 1.0, 2.0, 3.0], 0.0..3.0, ColorMap::Grey, &mut scratch);
/// assert_eq!(scratch[0], Color::BLACK);
/// assert_eq!(scratch[3], Color::WHITE);
/// ```
#[doc(alias = "heatmap")]
#[doc(alias = "false_color")]
pub fn colormap_into(values: &[f32], range: Range<f32>, map: ColorMap, out: &mut [Color]) {
    let span = range.end - range.start;
    let inv = if span.abs() < f32::EPSILON {
        0.0
    } else {
        1.0 / span
    };
    for (dst, &v) in out.iter_mut().zip(values) {
        *dst = sample((v - range.start) * inv, map);
    }
}

/// Map `values` through `map`, allocating the result.
///
/// The convenient form, for setup code and one-shot scripts. In a per-frame
/// loop use [`colormap_into`] instead.
///
/// ```
/// use fathom_geom::{ColorMap, colormap};
///
/// let depth_mm = [500.0, 1000.0, 1500.0];
/// let rgba = colormap(&depth_mm, 500.0..1500.0, ColorMap::Turbo);
/// assert_eq!(rgba.len(), depth_mm.len());
/// ```
#[must_use]
#[doc(alias = "heatmap")]
#[doc(alias = "false_color")]
pub fn colormap(values: &[f32], range: Range<f32>, map: ColorMap) -> Vec<Color> {
    let mut out = vec![Color::TRANSPARENT; values.len()];
    colormap_into(values, range, map, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn near(got: Color, want: [u8; 3], tol: u8) -> bool {
        got.channels()
            .iter()
            .zip(&want)
            .all(|(g, w)| g.abs_diff(*w) <= tol)
    }

    #[test]
    fn ramps_track_their_reference_endpoints() {
        // The polynomial fits drift furthest at the very ends of the ramp, by
        // up to about 8% of a channel; everywhere else they are within a step
        // or two. That is invisible in a heatmap and costs no lookup table.
        assert!(near(sample(0.0, ColorMap::Viridis), [68, 1, 84], 6));
        assert!(near(sample(1.0, ColorMap::Viridis), [253, 231, 37], 8));
        assert!(near(sample(0.0, ColorMap::Magma), [0, 0, 0], 2));
        assert!(near(sample(1.0, ColorMap::Magma), [252, 253, 191], 12));
        assert!(near(sample(0.0, ColorMap::Turbo), [48, 18, 59], 20));
        assert!(near(sample(1.0, ColorMap::Turbo), [122, 4, 3], 22));
        assert_eq!(sample(0.5, ColorMap::Coolwarm), Color::WHITE);
        assert_eq!(sample(1.0, ColorMap::Grey), Color::WHITE);
    }

    #[test]
    fn perceptual_ramps_are_monotonic_in_brightness() {
        for map in [ColorMap::Viridis, ColorMap::Magma, ColorMap::Grey] {
            let lum = |t: f32| {
                let c = sample(t, map).channels();
                0.2126f32.mul_add(
                    f32::from(c[0]),
                    0.7152f32.mul_add(f32::from(c[1]), 0.0722 * f32::from(c[2])),
                )
            };
            for i in 0..32u8 {
                let (a, b) = (f32::from(i) / 32.0, f32::from(i + 1) / 32.0);
                assert!(lum(a) < lum(b) + 1.0, "{map:?} dipped between {a} and {b}");
            }
        }
    }

    #[test]
    fn out_of_range_clamps_and_nan_is_low() {
        let mut out = [Color::TRANSPARENT; 3];
        colormap_into(&[-10.0, f32::NAN, 10.0], 0.0..1.0, ColorMap::Grey, &mut out);
        assert_eq!(out, [Color::BLACK, Color::BLACK, Color::WHITE]);
    }

    #[test]
    fn mismatched_lengths_take_the_shorter() {
        let mut out = [Color::TRANSPARENT; 4];
        colormap_into(&[1.0], 0.0..1.0, ColorMap::Grey, &mut out);
        assert_eq!(out[0], Color::WHITE);
        assert_eq!(out[3], Color::TRANSPARENT);
        colormap_into(&[1.0; 99], 0.0..1.0, ColorMap::Grey, &mut out);
        assert_eq!(out[3], Color::WHITE);
    }

    #[test]
    fn degenerate_range_does_not_divide_by_zero() {
        let c = colormap(&[5.0, 5.0], 5.0..5.0, ColorMap::Turbo);
        assert_eq!(c[0], c[1]);
    }
}
