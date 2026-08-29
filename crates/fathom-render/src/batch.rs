//! The vertex arena and the command list.
//!
//! Allocated at init, never during a frame. Draw calls append into the arena;
//! when it fills, the contents are handed to a GPU chunk buffer and the arena
//! keeps going from empty. Nothing is dropped and nothing is reallocated, so a
//! pathological frame degrades into a second vertex buffer rather than into a
//! missing trajectory.

use std::sync::Arc;

use crate::vertex::{Topology, Vertex};

/// Viewports are copied verbatim out of a `Rect`, never computed, so bitwise
/// equality is exactly the question being asked: did the caller hand us the
/// same panel again?
#[inline]
fn viewports_equal(a: [f32; 4], b: [f32; 4]) -> bool {
    a.iter().zip(&b).all(|(a, b)| a.to_bits() == b.to_bits())
}

/// One contiguous run of vertices sharing a pipeline and a texture.
#[derive(Debug)]
struct Cmd {
    topology: Topology,
    /// Pixel rectangle this run draws into: `[x, y, w, h]`. 2D uses the whole
    /// framebuffer; a `Scene` uses its panel, so a 3D view can sit beside a
    /// video without bleeding over it.
    viewport: [f32; 4],
    /// `None` means the font/white atlas, which is bound once and serves every
    /// untextured primitive through its solid white texel.
    texture: Option<Arc<wgpu::BindGroup>>,
    chunk: u32,
    start: u32,
    count: u32,
}

/// Packs draw calls into vertices and vertices into one submit.
#[derive(Debug)]
pub(crate) struct Batcher {
    arena: Vec<Vertex>,
    chunks: Vec<wgpu::Buffer>,
    cmds: Vec<Cmd>,
    chunk: u32,
    capacity: usize,
    peak: usize,
}

/// Vertices per arena, and so per GPU chunk buffer. 64k vertices is 1.75MB,
/// which holds a 10k-point cloud plus a full HUD without ever reaching for a
/// second chunk.
pub(crate) const DEFAULT_ARENA: usize = 64 * 1024;

impl Batcher {
    pub(crate) fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1024);
        Self {
            arena: Vec::with_capacity(capacity),
            chunks: Vec::with_capacity(2),
            cmds: Vec::with_capacity(256),
            chunk: 0,
            capacity,
            peak: 0,
        }
    }

    /// How many vertices fit in one batch. Primitives longer than this split
    /// themselves across several, which is why no draw call has a size limit.
    pub(crate) const fn batch_limit(&self) -> usize {
        self.capacity
    }

    /// The most vertices any one frame has packed so far, for `Ctx::peak_vertices`.
    pub(crate) const fn peak(&self) -> usize {
        self.peak
    }

    pub(crate) fn begin_frame(&mut self) {
        self.arena.clear();
        self.cmds.clear();
        self.chunk = 0;
        self.peak = 0;
    }

    /// Make room for `count` more vertices and open a batch to receive them.
    ///
    /// Merges into the previous command when the pipeline, the texture and the
    /// chunk all match, so a hundred `draw_line_3d` calls in a row cost one
    /// draw call, not a hundred.
    pub(crate) fn begin(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        topology: Topology,
        texture: Option<&Arc<wgpu::BindGroup>>,
        viewport: [f32; 4],
        count: usize,
    ) {
        debug_assert!(
            count <= self.capacity,
            "primitive must be split by its caller"
        );
        if self.arena.len() + count > self.capacity {
            self.flush(device, queue);
        }

        #[allow(clippy::cast_possible_truncation)]
        let start = self.arena.len() as u32;
        let mergeable = self.cmds.last().is_some_and(|last| {
            last.topology == topology
                && viewports_equal(last.viewport, viewport)
                && last.chunk == self.chunk
                && last.start + last.count == start
                && match (&last.texture, texture) {
                    (None, None) => true,
                    (Some(a), Some(b)) => Arc::ptr_eq(a, b),
                    _ => false,
                }
        });

        #[allow(clippy::cast_possible_truncation)]
        let count = count as u32;
        if mergeable {
            if let Some(last) = self.cmds.last_mut() {
                last.count += count;
            }
        } else {
            self.cmds.push(Cmd {
                topology,
                viewport,
                texture: texture.cloned(),
                chunk: self.chunk,
                start,
                count,
            });
        }
    }

    /// Append one vertex. [`Batcher::begin`] has already guaranteed the room,
    /// so this never reallocates and never reaches the GPU on its own.
    #[inline]
    pub(crate) fn push(&mut self, v: Vertex) {
        debug_assert!(self.arena.len() < self.capacity, "begin() under-reserved");
        self.arena.push(v);
    }

    /// Hand the filled arena to a chunk buffer and continue from empty.
    fn flush(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.arena.is_empty() {
            return;
        }
        self.peak += self.arena.len();

        let bytes: &[u8] = bytemuck::cast_slice(&self.arena);
        if self.chunks.len() <= self.chunk as usize {
            // Only ever on the frame that first overflows; afterwards the pool
            // is warm and this branch is dead for the rest of the session.
            self.chunks
                .push(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("fathom.vertices"),
                    size: (self.capacity * core::mem::size_of::<Vertex>()) as wgpu::BufferAddress,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
        }
        if let Some(buffer) = self.chunks.get(self.chunk as usize) {
            queue.write_buffer(buffer, 0, bytes);
        }
        self.chunk += 1;
        self.arena.clear();
    }

    /// Flush the tail, then replay every command into one render pass.
    pub(crate) fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pass: &mut wgpu::RenderPass<'_>,
        pipelines: &crate::ctx::Pipelines,
        atlas: &wgpu::BindGroup,
    ) {
        self.flush(device, queue);

        let mut bound_chunk = None;
        let mut bound_topology = None;
        let mut bound_texture: Option<*const wgpu::BindGroup> = None;
        let mut bound_viewport = None;

        for cmd in &self.cmds {
            let Some(buffer) = self.chunks.get(cmd.chunk as usize) else {
                continue;
            };
            if bound_chunk != Some(cmd.chunk) {
                pass.set_vertex_buffer(0, buffer.slice(..));
                bound_chunk = Some(cmd.chunk);
            }
            if bound_viewport != Some(cmd.viewport) {
                let [x, y, w, h] = cmd.viewport;
                pass.set_viewport(x, y, w.max(1.0), h.max(1.0), 0.0, 1.0);
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                pass.set_scissor_rect(x as u32, y as u32, w.max(1.0) as u32, h.max(1.0) as u32);
                bound_viewport = Some(cmd.viewport);
            }
            if bound_topology != Some(cmd.topology) {
                pass.set_pipeline(pipelines.get(cmd.topology));
                bound_topology = Some(cmd.topology);
            }
            let group = cmd.texture.as_deref().unwrap_or(atlas);
            if bound_texture != Some(core::ptr::from_ref(group)) {
                pass.set_bind_group(0, group, &[]);
                bound_texture = Some(core::ptr::from_ref(group));
            }
            pass.draw(cmd.start..cmd.start + cmd.count, 0..1);
        }
    }
}
