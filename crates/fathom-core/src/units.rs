//! Unit newtypes and screen rectangles.
//!
//! Free at runtime, and they catch the class of bug that actually bites in a
//! robotics cell: metres passed where radians were meant, a frame index passed
//! where a timestamp was meant.

use core::{
    fmt,
    ops::{Add, Div, Mul, Neg, Sub},
};

macro_rules! scalar_newtype {
    ($(#[$m:meta])* $name:ident, $inner:ty, $zero:expr, $unit:literal) => {
        $(#[$m])*
        #[repr(transparent)]
        #[derive(Clone, Copy, Default, PartialEq, PartialOrd, bytemuck::Pod, bytemuck::Zeroable)]
        pub struct $name(pub $inner);

        impl $name {
            /// The zero value.
            pub const ZERO: Self = Self($zero);

            /// Unwrap to the raw scalar.
            #[inline]
            #[must_use]
            pub const fn get(self) -> $inner {
                self.0
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{:?}{}", self.0, $unit)
            }
        }

        impl Add for $name {
            type Output = Self;
            #[inline]
            fn add(self, rhs: Self) -> Self { Self(self.0 + rhs.0) }
        }
        impl Sub for $name {
            type Output = Self;
            #[inline]
            fn sub(self, rhs: Self) -> Self { Self(self.0 - rhs.0) }
        }
        impl Neg for $name {
            type Output = Self;
            #[inline]
            fn neg(self) -> Self { Self(-self.0) }
        }
        impl Mul<$inner> for $name {
            type Output = Self;
            #[inline]
            fn mul(self, rhs: $inner) -> Self { Self(self.0 * rhs) }
        }
        impl Div<$inner> for $name {
            type Output = Self;
            #[inline]
            fn div(self, rhs: $inner) -> Self { Self(self.0 / rhs) }
        }
    };
}

scalar_newtype!(
    /// A length in metres. The only length unit fathom speaks.
    ///
    /// ```
    /// use fathom_core::Meters;
    /// assert_eq!((Meters(0.3) + Meters(0.2)).get(), 0.5);
    /// ```
    Meters,
    f32,
    0.0,
    "m"
);

scalar_newtype!(
    /// An angle in radians. The only angle unit fathom speaks.
    Radians,
    f32,
    0.0,
    "rad"
);

scalar_newtype!(
    /// A wall-clock instant in **nanoseconds**, the axis every stream shares.
    ///
    /// Video arrives at 30Hz, joint states at 500Hz, a language instruction once
    /// per episode: a frame index cannot express that, so fathom never uses one
    /// as a time axis. Alignment is caller code, and it is one line:
    ///
    /// ```
    /// use fathom_core::Timestamp;
    ///
    /// let states = [Timestamp(0), Timestamp(2_000_000), Timestamp(4_000_000)];
    /// let now = Timestamp(3_000_000);
    /// let i = states.partition_point(|t| *t <= now) - 1;
    /// assert_eq!(i, 1);
    /// ```
    Timestamp,
    i64,
    0,
    "ns"
);

impl Timestamp {
    /// Build a timestamp from seconds.
    #[inline]
    #[must_use]
    pub fn from_secs_f64(s: f64) -> Self {
        #[allow(clippy::cast_possible_truncation)]
        Self((s * 1e9) as i64)
    }

    /// This instant in seconds, for display.
    #[inline]
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn as_secs_f64(self) -> f64 {
        self.0 as f64 / 1e9
    }
}

/// An index into a decoded stream. Never a time axis; see [`Timestamp`].
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FrameIdx(pub u64);

/// Integer text magnification, so a bitmap glyph stays crisp.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum TextScale {
    /// 8px cell.
    #[default]
    X1,
    /// 16px cell.
    X2,
    /// 24px cell.
    X3,
    /// 32px cell.
    X4,
}

impl TextScale {
    /// The magnification factor as a number of pixels per glyph pixel.
    #[inline]
    #[must_use]
    pub const fn factor(self) -> f32 {
        match self {
            Self::X1 => 1.0,
            Self::X2 => 2.0,
            Self::X3 => 3.0,
            Self::X4 => 4.0,
        }
    }
}

/// An axis-aligned rectangle in pixels, origin at the top-left corner.
///
/// Multi-panel layout is `Rect` math in your code; fathom ships no layout
/// manager:
///
/// ```
/// use fathom_core::Rect;
///
/// let window = Rect::new(0.0, 0.0, 1280.0, 720.0);
/// let [left, right] = window.split_h();
/// assert_eq!(right.x, 640.0);
/// assert_eq!(left.w, 640.0);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Rect {
    /// Left edge.
    pub x: f32,
    /// Top edge.
    pub y: f32,
    /// Width, extending right.
    pub w: f32,
    /// Height, extending down.
    pub h: f32,
}

impl Rect {
    /// Build a rectangle from its top-left corner and size.
    #[inline]
    #[must_use]
    pub const fn new(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self { x, y, w, h }
    }

    /// Right edge.
    #[inline]
    #[must_use]
    pub fn right(self) -> f32 {
        self.x + self.w
    }

    /// Bottom edge.
    #[inline]
    #[must_use]
    pub fn bottom(self) -> f32 {
        self.y + self.h
    }

    /// Is this pixel inside the rectangle?
    #[inline]
    #[must_use]
    pub fn contains(self, x: f32, y: f32) -> bool {
        x >= self.x && y >= self.y && x < self.right() && y < self.bottom()
    }

    /// Shrink by `pad` on every side, clamped to non-negative size.
    #[inline]
    #[must_use]
    pub fn inset(self, pad: f32) -> Self {
        Self::new(
            self.x + pad,
            self.y + pad,
            (self.w - 2.0 * pad).max(0.0),
            (self.h - 2.0 * pad).max(0.0),
        )
    }

    /// Split into left and right halves.
    #[inline]
    #[must_use]
    pub fn split_h(self) -> [Self; 2] {
        let w = self.w * 0.5;
        [
            Self::new(self.x, self.y, w, self.h),
            Self::new(self.x + w, self.y, w, self.h),
        ]
    }

    /// Split into top and bottom halves.
    #[inline]
    #[must_use]
    pub fn split_v(self) -> [Self; 2] {
        let h = self.h * 0.5;
        [
            Self::new(self.x, self.y, self.w, h),
            Self::new(self.x, self.y + h, self.w, h),
        ]
    }

    /// The largest rectangle of the given aspect ratio centred inside this one.
    ///
    /// This is what keeps a 4:3 camera stream from stretching in a 16:9 panel.
    #[inline]
    #[must_use]
    pub fn fit_aspect(self, aspect: f32) -> Self {
        if aspect <= 0.0 || self.h <= 0.0 {
            return self;
        }
        let (w, h) = if self.w / self.h > aspect {
            (self.h * aspect, self.h)
        } else {
            (self.w, self.w / aspect)
        };
        Self::new(
            self.x + (self.w - w) * 0.5,
            self.y + (self.h - h) * 0.5,
            w,
            h,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rect_math() {
        let r = Rect::new(0.0, 0.0, 16.0, 9.0);
        assert!(r.contains(15.9, 8.9));
        assert!(!r.contains(16.0, 0.0));
        assert_eq!(r.inset(20.0), Rect::new(20.0, 20.0, 0.0, 0.0));

        let fit = Rect::new(0.0, 0.0, 100.0, 100.0).fit_aspect(2.0);
        assert_eq!(fit, Rect::new(0.0, 25.0, 100.0, 50.0));
    }

    #[test]
    fn timestamp_roundtrip() {
        assert_eq!(Timestamp::from_secs_f64(1.5), Timestamp(1_500_000_000));
        assert!((Timestamp(1_500_000_000).as_secs_f64() - 1.5).abs() < 1e-12);
    }
}
