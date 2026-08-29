//! Packed RGBA color and the colormap selector.

use core::fmt;

/// A non-premultiplied sRGB color, stored as `[r, g, b, a]`.
///
/// `#[repr(transparent)]` over `[u8; 4]` rather than over a `u32`, and that
/// choice is load-bearing: it means `bytemuck::cast_slice(&colors)` yields
/// bytes in red-green-blue-alpha order on every target, so the output of
/// [`colormap`](../fathom_geom/fn.colormap.html) can be handed straight to
/// `upload_texture`. Packing into a `u32` would put the channels in memory
/// backwards on every little-endian machine, which is to say all of them, and
/// the symptom would be a heatmap with red and blue swapped.
///
/// Use [`Color::hex`] when a literal reads better.
///
/// ```
/// use fathom_core::Color;
///
/// assert_eq!(Color::rgb(255, 0, 0), Color::RED);
/// assert_eq!(Color::hex(0xFF_00_00_FF), Color::RED);
/// assert_eq!(Color::RED.with_alpha(0.5).channels(), [255, 0, 0, 128]);
///
/// // The reason for the layout: bytes come out in the order a texture wants.
/// let colors = [Color::RED, Color::GREEN];
/// assert_eq!(bytemuck::cast_slice::<Color, u8>(&colors), &[255, 0, 0, 255, 0, 255, 0, 255]);
/// ```
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Color(pub [u8; 4]);

impl Color {
    /// Fully transparent.
    pub const TRANSPARENT: Self = Self([0, 0, 0, 0]);
    /// Opaque black.
    pub const BLACK: Self = Self::rgb(0, 0, 0);
    /// Opaque white.
    pub const WHITE: Self = Self::rgb(255, 255, 255);
    /// Opaque red.
    pub const RED: Self = Self::rgb(255, 0, 0);
    /// Opaque green.
    pub const GREEN: Self = Self::rgb(0, 255, 0);
    /// Opaque blue.
    pub const BLUE: Self = Self::rgb(0, 0, 255);
    /// Opaque yellow.
    pub const YELLOW: Self = Self::rgb(255, 255, 0);
    /// Opaque magenta.
    pub const MAGENTA: Self = Self::rgb(255, 0, 255);
    /// Opaque cyan.
    pub const CYAN: Self = Self::rgb(0, 255, 255);
    /// Opaque mid grey.
    pub const GRAY: Self = Self::rgb(128, 128, 128);

    /// Build an opaque color from 8-bit channels.
    #[inline]
    #[must_use]
    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self::rgba(r, g, b, 255)
    }

    /// Build a color from 8-bit channels including alpha.
    #[inline]
    #[must_use]
    pub const fn rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self([r, g, b, a])
    }

    /// Build a color from a `0xRRGGBBAA` literal.
    #[inline]
    #[must_use]
    pub const fn hex(v: u32) -> Self {
        Self(v.to_be_bytes())
    }

    /// The channels as `[r, g, b, a]`, which is also the memory layout.
    #[inline]
    #[must_use]
    pub const fn channels(self) -> [u8; 4] {
        self.0
    }

    /// Repack into a `0xRRGGBBAA` integer, for display or hashing.
    #[inline]
    #[must_use]
    pub const fn to_hex(self) -> u32 {
        u32::from_be_bytes(self.0)
    }

    /// Return this color with alpha replaced, `t` clamped to `0.0..=1.0`.
    #[inline]
    #[must_use]
    pub fn with_alpha(self, a: f32) -> Self {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let a = (a.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        let [r, g, b, _] = self.0;
        Self([r, g, b, a])
    }

    /// Linearly interpolate in 8-bit sRGB space, `t` clamped to `0.0..=1.0`.
    #[inline]
    #[must_use]
    pub fn lerp(self, other: Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        let mut out = [0u8; 4];
        for (dst, (lo, hi)) in out
            .iter_mut()
            .zip(self.channels().into_iter().zip(other.channels()))
        {
            let (lo, hi) = (f32::from(lo), f32::from(hi));
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            {
                *dst = (hi.mul_add(t, lo * (1.0 - t)) + 0.5) as u8;
            }
        }
        Self::from(out)
    }
}

impl fmt::Debug for Color {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{:08X}", self.to_hex())
    }
}

impl From<u32> for Color {
    /// From a `0xRRGGBBAA` literal; see [`Color::hex`].
    #[inline]
    fn from(v: u32) -> Self {
        Self::hex(v)
    }
}

impl From<[u8; 4]> for Color {
    #[inline]
    fn from(v: [u8; 4]) -> Self {
        Self::rgba(v[0], v[1], v[2], v[3])
    }
}

/// Which perceptual ramp `colormap` maps scalars through.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[doc(alias = "heatmap")]
#[doc(alias = "palette")]
pub enum ColorMap {
    /// Google's Turbo: rainbow-like but perceptually ordered. Good default for depth.
    #[default]
    Turbo,
    /// Matplotlib's Viridis: perceptually uniform, colorblind-safe.
    Viridis,
    /// Matplotlib's Magma: perceptually uniform, dark to bright.
    Magma,
    /// Linear greyscale.
    Grey,
    /// Blue-white-red, for signed quantities read around the middle of the range.
    Coolwarm,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channel_order_is_rgba_in_memory() {
        assert_eq!(Color::rgba(1, 2, 3, 4).channels(), [1, 2, 3, 4]);
        // The whole reason for the layout: what a texture upload will see.
        assert_eq!(
            bytemuck::cast_slice::<Color, u8>(&[Color::rgba(1, 2, 3, 4), Color::rgba(5, 6, 7, 8)]),
            &[1, 2, 3, 4, 5, 6, 7, 8]
        );
        assert_eq!(Color::hex(0x01_02_03_04), Color::rgba(1, 2, 3, 4));
        assert_eq!(Color::rgba(1, 2, 3, 4).to_hex(), 0x01_02_03_04);
        assert_eq!(
            Color::BLACK.lerp(Color::WHITE, 0.5).channels(),
            [128, 128, 128, 255]
        );
        assert_eq!(Color::BLACK.lerp(Color::WHITE, 2.0), Color::WHITE);
    }
}
