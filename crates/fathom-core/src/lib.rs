//! Shared types for [fathom](https://docs.rs/fathom), a low-level visualization
//! library for multimodal spatiotemporal data.
//!
//! This crate is deliberately tiny and rendering-free: it is the only thing the
//! adapter ecosystem (MCAP, LeRobot, ROS bags, video containers) has to agree
//! on. An adapter is just a crate whose functions return slices of these types.
//!
//! ```
//! use fathom_core::{Color, Meters, Point, World};
//!
//! let path = [Point::<World>::new(0.0, 0.0, 0.0), Point::<World>::new(1.0, 0.0, 0.0)];
//! let length = Meters(path[0].0.distance(path[1].0));
//! assert_eq!(length, Meters(1.0));
//! assert_eq!(Color::RED.channels(), [255, 0, 0, 255]);
//! ```
#![no_std]
#![deny(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

mod calib;
mod color;
mod point;
mod units;

pub use calib::{CalibError, Extrinsics, Homography, Intrinsics, look_at};
pub use color::{Color, ColorMap};
pub use glam::{Mat3, Mat4, Quat, Vec2, Vec3};
pub use point::{Camera, CoordFrame, Image, Plane, Point, World, point};
pub use units::{FrameIdx, Meters, Radians, Rect, TextScale, Timestamp};
