//! The one vertex format, shared by both pipelines.

use bytemuck::{Pod, Zeroable};
use fathom_core::Color;
use glam::{Vec2, Vec4};

/// A packed vertex: clip-space position, atlas coordinate, color.
///
/// 28 bytes, `#[repr(C)]`, `Pod`. The arena is a `&[Vertex]` that reaches the
/// GPU through one `bytemuck::cast_slice`, with no serialization step.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct Vertex {
    /// Position in clip space; the perspective divide is the GPU's job.
    pub clip: [f32; 4],
    /// Texture coordinate into the bound texture, `0.0..=1.0`.
    pub uv: [f32; 2],
    /// Non-premultiplied RGBA, normalized by the vertex format.
    pub color: [u8; 4],
}

impl Vertex {
    /// Build a vertex from a clip-space position, a UV and a color.
    #[inline]
    #[must_use]
    pub fn new(clip: Vec4, uv: Vec2, color: Color) -> Self {
        Self {
            clip: clip.to_array(),
            uv: uv.to_array(),
            color: color.channels(),
        }
    }

    pub(crate) const ATTRS: [wgpu::VertexAttribute; 3] = wgpu::vertex_attr_array![
        0 => Float32x4,
        1 => Float32x2,
        2 => Unorm8x4,
    ];

    pub(crate) const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: core::mem::size_of::<Self>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &Self::ATTRS,
    };
}

/// Which of the two pipelines a batch of vertices belongs to.
///
/// Every primitive in the library lowers to one of these. Adding a primitive
/// means writing a lowering function, never a new shader path.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Topology {
    /// Textured triangles: quads, glyphs, filled polygons, billboarded points.
    Triangles,
    /// Untextured line segments: trajectories, grids, axis triads, wireframes.
    Lines,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vertex_is_tightly_packed() {
        assert_eq!(core::mem::size_of::<Vertex>(), 28);
        assert_eq!(Vertex::LAYOUT.array_stride, 28);
    }
}
