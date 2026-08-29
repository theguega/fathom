//! Packed RGBA color and the colormap selector.

use core::fmt;

/// A packed, non-premultiplied sRGB color: `0xRRGGBBAA`.
///
/// `#[repr(transparent)]` over `u32`, so a `&[Color]` uploads to the GPU with
/// no conversion step.
///
/// ```
/// use fathom_core::Color;
///
/// assert_eq!(Color::rgb(255, 0, 0), Color::RED);
/// assert_eq!(Color::RED.with_alpha(0.5).channels(), [255, 0, 0, 128]);
/// ```
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Color(pub u32);

impl Color {
    /// Fully transparent.
    pub const TRANSPARENT: Self = Self(0x0000_0000);
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
        Self(u32::from_be_bytes([r, g, b, a]))
    }

    /// Unpack to `[r, g, b, a]`, the byte order the vertex format expects.
    #[inline]
    #[must_use]
    pub const fn channels(self) -> [u8; 4] {
        self.0.to_be_bytes()
    }

    /// Return this color with alpha replaced, `t` clamped to `0.0..=1.0`.
    #[inline]
    #[must_use]
    pub fn with_alpha(self, a: f32) -> Self {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let a = (a.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        Self((self.0 & 0xFFFF_FF00) | u32::from(a))
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
        write!(f, "#{:08X}", self.0)
    }
}

impl From<u32> for Color {
    #[inline]
    fn from(v: u32) -> Self {
        Self(v)
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
        assert_eq!(
            Color::BLACK.lerp(Color::WHITE, 0.5).channels(),
            [128, 128, 128, 255]
        );
        assert_eq!(Color::BLACK.lerp(Color::WHITE, 2.0), Color::WHITE);
    }
}
