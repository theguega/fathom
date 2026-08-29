//! Coordinate-framed points.

use core::{fmt, marker::PhantomData};

use bytemuck::Pod;
use glam::{Vec2, Vec3};

use crate::pod::pod;

/// A coordinate frame marker.
///
/// Implement this on a zero-sized type to add a frame of your own; the marker
/// never exists at runtime, it only makes [`Point`] values of different frames
/// refuse to mix.
///
/// ```
/// use fathom_core::{CoordFrame, Point, Vec3};
///
/// #[derive(Clone, Copy, Debug)]
/// struct Tool;
/// impl CoordFrame for Tool {
///     type Repr = Vec3;
/// }
///
/// let tip = Point::<Tool>::from_repr(Vec3::new(0.0, 0.0, 0.1));
/// assert_eq!(tip.0.z, 0.1);
/// ```
pub trait CoordFrame: Copy + fmt::Debug + 'static {
    /// The vector this frame's points are stored as: [`Vec2`] or [`Vec3`].
    type Repr: Copy + fmt::Debug + PartialEq + Pod;
}

/// Metric 3D world frame, the frame every [`Scene`](../fathom/struct.Scene.html) draw call speaks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct World;

/// Metric 3D camera frame: +X right, +Y down, +Z forward along the optical axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Camera;

/// 2D pixel frame of an image or of the window, origin at the top-left corner.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Image;

/// 2D frame of a physical plane, such as a workcell floor viewed by a fixed camera.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Plane;

impl CoordFrame for World {
    type Repr = Vec3;
}
impl CoordFrame for Camera {
    type Repr = Vec3;
}
impl CoordFrame for Image {
    type Repr = Vec2;
}
impl CoordFrame for Plane {
    type Repr = Vec2;
}

/// A point tagged with the coordinate frame it lives in.
///
/// `#[repr(transparent)]` over [`Vec2`] or [`Vec3`], so the tag is free at
/// runtime and a `&[Point<World>]` is bit-identical to a `&[Vec3]`. The tag is
/// what turns an extrinsics mix-up into a compile error:
///
/// ```compile_fail
/// use fathom_core::{Camera, Point, World};
///
/// fn takes_world(p: Point<World>) {}
/// takes_world(Point::<Camera>::new(1.0, 2.0, 3.0)); // wrong frame, does not compile
/// ```
#[repr(transparent)]
pub struct Point<F: CoordFrame>(pub F::Repr, PhantomData<fn() -> F>);

impl<F: CoordFrame> Point<F> {
    /// Wrap a raw vector as a point in frame `F`.
    #[inline]
    #[must_use]
    pub const fn from_repr(v: F::Repr) -> Self {
        Self(v, PhantomData)
    }

    /// Unwrap to the raw vector, discarding the frame tag.
    #[inline]
    #[must_use]
    pub const fn into_repr(self) -> F::Repr {
        self.0
    }
}

/// Construct a point in frame `F` from its raw vector, as a free function.
///
/// Shorthand for [`Point::from_repr`], useful in iterator chains:
///
/// ```
/// use fathom_core::{Point, World, point, Vec3};
///
/// let pts: Vec<Point<World>> = [Vec3::ZERO, Vec3::X].into_iter().map(point).collect();
/// assert_eq!(pts.len(), 2);
/// ```
#[inline]
#[must_use]
pub const fn point<F: CoordFrame>(v: F::Repr) -> Point<F> {
    Point::from_repr(v)
}

/// A point in the metric world frame.
pub type WorldPoint = Point<World>;
/// A point in a camera's own frame.
pub type CameraPoint = Point<Camera>;
/// A pixel in an image or in the window.
pub type ImagePoint = Point<Image>;
/// A point on a physical plane.
pub type PlanePoint = Point<Plane>;

macro_rules! impl_3d {
    ($($f:ty),*) => {$(
        impl Point<$f> {
            /// Build a point from metric x, y, z.
            #[inline]
            #[must_use]
            pub const fn new(x: f32, y: f32, z: f32) -> Self {
                Self(Vec3::new(x, y, z), PhantomData)
            }
            /// The zero point of this frame.
            pub const ORIGIN: Self = Self::new(0.0, 0.0, 0.0);
        }

        pod!(Point<$f>, 12);
    )*};
}

macro_rules! impl_2d {
    ($($f:ty),*) => {$(
        impl Point<$f> {
            /// Build a point from x, y.
            #[inline]
            #[must_use]
            pub const fn new(x: f32, y: f32) -> Self {
                Self(Vec2::new(x, y), PhantomData)
            }
            /// The zero point of this frame.
            pub const ORIGIN: Self = Self::new(0.0, 0.0);
        }

        pod!(Point<$f>, 8);
    )*};
}

impl_3d!(World, Camera);
impl_2d!(Image, Plane);

#[allow(clippy::expl_impl_clone_on_copy)] // derive would bound `F: Clone`, which the tag does not need
impl<F: CoordFrame> Clone for Point<F> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<F: CoordFrame> Copy for Point<F> {}

impl<F: CoordFrame> PartialEq for Point<F> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<F: CoordFrame> fmt::Debug for Point<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Point<{}>{:?}", core::any::type_name::<F>(), self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transparent_over_its_vector() {
        assert_eq!(
            core::mem::size_of::<Point<World>>(),
            core::mem::size_of::<Vec3>()
        );
        assert_eq!(
            core::mem::size_of::<Point<Image>>(),
            core::mem::size_of::<Vec2>()
        );
        let raw = [Vec3::X, Vec3::Y];
        let pts: &[Point<World>] = bytemuck::cast_slice(&raw);
        assert_eq!(
            pts,
            [
                Point::<World>::new(1.0, 0.0, 0.0),
                Point::<World>::new(0.0, 1.0, 0.0)
            ]
        );
    }
}
