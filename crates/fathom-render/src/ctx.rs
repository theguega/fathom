//! The GPU context: created once, at the boundary, and fallible exactly there.

use std::{fmt, sync::Arc};

use fathom_core::Color;

use crate::{
    batch::{Batcher, DEFAULT_ARENA},
    font,
    texture::Filter,
    vertex::{Topology, Vertex},
};

/// Why a context could not be created.
///
/// Every variant is a genuine boundary: a missing GPU, a driver that refuses
/// the surface, a shader the backend will not accept. Once [`Ctx::new`]
/// returns, drawing is infallible.
#[derive(Debug)]
#[non_exhaustive]
#[allow(clippy::enum_variant_names)] // every one of them is an absence; the prefix is the point
pub enum InitError {
    /// No graphics adapter matched the request.
    NoAdapter(wgpu::RequestAdapterError),
    /// An adapter was found but would not hand over a device.
    NoDevice(wgpu::RequestDeviceError),
    /// The window handle could not be turned into a drawable surface.
    NoSurface(wgpu::CreateSurfaceError),
    /// The adapter supports no texture format this renderer can present to.
    NoSurfaceFormat,
}

impl fmt::Display for InitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoAdapter(_) => f.write_str("no graphics adapter available"),
            Self::NoDevice(_) => f.write_str("adapter would not provide a device"),
            Self::NoSurface(_) => f.write_str("could not create a surface for this window"),
            Self::NoSurfaceFormat => f.write_str("adapter presents no supported texture format"),
        }
    }
}

impl std::error::Error for InitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::NoAdapter(e) => Some(e),
            Self::NoDevice(e) => Some(e),
            Self::NoSurface(e) => Some(e),
            Self::NoSurfaceFormat => None,
        }
    }
}

/// Where a frame ends up: a window, or a texture you can read back.
///
/// Both are the same draw code. If an example has to change between them, the
/// API leaked a display assumption, which is the whole point of the headless
/// path existing.
enum Output {
    Surface {
        surface: wgpu::Surface<'static>,
        config: wgpu::SurfaceConfiguration,
    },
    Offscreen {
        texture: wgpu::Texture,
        view: wgpu::TextureView,
    },
}

impl fmt::Debug for Output {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Surface { .. } => f.write_str("Surface"),
            Self::Offscreen { .. } => f.write_str("Offscreen"),
        }
    }
}

/// The two pipelines. Every primitive in the library lowers to one of them.
#[derive(Debug)]
pub(crate) struct Pipelines {
    triangles: wgpu::RenderPipeline,
    lines: wgpu::RenderPipeline,
}

impl Pipelines {
    pub(crate) const fn get(&self, topology: Topology) -> &wgpu::RenderPipeline {
        match topology {
            Topology::Triangles => &self.triangles,
            Topology::Lines => &self.lines,
        }
    }
}

/// The renderer. One per window, created once.
///
/// Holds the device, the two pipelines, the glyph atlas and the vertex arena.
/// Nothing here is allocated again once [`Ctx::new`] returns, which is what the
/// per-frame allocation test in the benches pins down.
#[derive(Debug)]
pub struct Ctx {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    output: Output,
    layout: wgpu::BindGroupLayout,
    pipelines: Pipelines,
    atlas: wgpu::BindGroup,
    depth: wgpu::TextureView,
    pub(crate) batcher: Batcher,
    clear: Color,
    size: (u32, u32),
}

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

impl Ctx {
    /// Create a context that draws into a window.
    ///
    /// Takes anything wgpu can turn into a surface, which in practice is an
    /// `Arc<winit::window::Window>`. The `Arc` is what keeps the surface sound
    /// without `unsafe` and without a lifetime on [`Ctx`].
    ///
    /// # Errors
    ///
    /// See [`InitError`]. All of them mean "this machine cannot draw", and all
    /// of them happen here rather than somewhere inside a frame.
    pub fn new(
        window: impl Into<wgpu::SurfaceTarget<'static>>,
        width: u32,
        height: u32,
    ) -> Result<Self, InitError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let surface = instance
            .create_surface(window)
            .map_err(InitError::NoSurface)?;
        let (device, queue, adapter) = Self::request_device(&instance, Some(&surface))?;

        let caps = surface.get_capabilities(&adapter);
        // Prefer a non-sRGB format: `Color` is already sRGB bytes, so letting
        // the hardware convert again would wash every color out.
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
            .or_else(|| caps.formats.first().copied())
            .ok_or(InitError::NoSurfaceFormat)?;

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: width.max(1),
            height: height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps
                .alpha_modes
                .first()
                .copied()
                .unwrap_or(wgpu::CompositeAlphaMode::Auto),
            view_formats: Vec::with_capacity(0),
        };
        surface.configure(&device, &config);

        Self::assemble(
            device,
            queue,
            Output::Surface { surface, config },
            format,
            (width.max(1), height.max(1)),
        )
    }

    /// Create a context that draws into an offscreen texture.
    ///
    /// The same draw calls as [`Ctx::new`], with no window and no display
    /// server, which is what makes rendering testable and what the mp4 export
    /// path is built on. Read the result back with [`Ctx::read_pixels`].
    ///
    /// # Errors
    ///
    /// See [`InitError`].
    pub fn headless(width: u32, height: u32) -> Result<Self, InitError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let (device, queue, _) = Self::request_device(&instance, None)?;

        let format = wgpu::TextureFormat::Rgba8Unorm;
        let (width, height) = (width.max(1), height.max(1));
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("fathom.offscreen"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        Self::assemble(
            device,
            queue,
            Output::Offscreen { texture, view },
            format,
            (width, height),
        )
    }

    fn request_device(
        instance: &wgpu::Instance,
        surface: Option<&wgpu::Surface<'static>>,
    ) -> Result<(Arc<wgpu::Device>, Arc<wgpu::Queue>, wgpu::Adapter), InitError> {
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            force_fallback_adapter: false,
            compatible_surface: surface,
        }))
        .map_err(InitError::NoAdapter)?;

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("fathom"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults().using_resolution(adapter.limits()),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .map_err(InitError::NoDevice)?;

        Ok((Arc::new(device), Arc::new(queue), adapter))
    }

    #[allow(clippy::unnecessary_wraps)] // fallible once more backends land; keeps the boundary honest
    fn assemble(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        output: Output,
        format: wgpu::TextureFormat,
        size: (u32, u32),
    ) -> Result<Self, InitError> {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fathom.texture"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fathom.shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fathom.pipeline"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let pipeline = |topology, label| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[Vertex::LAYOUT],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            })
        };

        let pipelines = Pipelines {
            triangles: pipeline(wgpu::PrimitiveTopology::TriangleList, "fathom.triangles"),
            lines: pipeline(wgpu::PrimitiveTopology::LineList, "fathom.lines"),
        };

        let atlas = Self::bake_atlas(&device, &queue, &layout);
        let depth = Self::depth_view(&device, size);

        Ok(Self {
            device,
            queue,
            output,
            layout,
            pipelines,
            atlas,
            depth,
            batcher: Batcher::new(DEFAULT_ARENA),
            clear: Color::rgb(16, 16, 20),
            size,
        })
    }

    fn bake_atlas(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
    ) -> wgpu::BindGroup {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("fathom.atlas"),
            size: wgpu::Extent3d {
                width: font::ATLAS,
                height: font::ATLAS,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &font::bake_atlas(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(font::ATLAS * 4),
                rows_per_image: Some(font::ATLAS),
            },
            wgpu::Extent3d {
                width: font::ATLAS,
                height: font::ATLAS,
                depth_or_array_layers: 1,
            },
        );
        Self::bind(device, layout, &texture, Filter::Nearest, "fathom.atlas")
    }

    pub(crate) fn bind(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        texture: &wgpu::Texture,
        filter: Filter,
        label: &str,
    ) -> wgpu::BindGroup {
        let mode = match filter {
            Filter::Linear => wgpu::FilterMode::Linear,
            Filter::Nearest => wgpu::FilterMode::Nearest,
        };
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some(label),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: mode,
            min_filter: mode,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &texture.create_view(&wgpu::TextureViewDescriptor::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        })
    }

    fn depth_view(device: &wgpu::Device, (width, height): (u32, u32)) -> wgpu::TextureView {
        device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("fathom.depth"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: DEPTH_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default())
    }

    /// Resize the drawing surface. Call it from your window's resize event.
    pub fn resize(&mut self, width: u32, height: u32) {
        let (width, height) = (width.max(1), height.max(1));
        if self.size == (width, height) {
            return;
        }
        self.size = (width, height);
        if let Output::Surface { surface, config } = &mut self.output {
            config.width = width;
            config.height = height;
            surface.configure(&self.device, config);
        }
        self.depth = Self::depth_view(&self.device, self.size);
    }

    /// The color the next frame is cleared to.
    pub fn set_clear_color(&mut self, color: Color) {
        self.clear = color;
    }

    /// Current drawing size in pixels.
    #[must_use]
    pub const fn size(&self) -> (u32, u32) {
        self.size
    }

    /// Width divided by height, ready for [`Camera::perspective`](crate::Camera::perspective).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn aspect(&self) -> f32 {
        self.size.0 as f32 / self.size.1 as f32
    }

    /// Vertices packed by the most recent frame.
    ///
    /// The number the fixed budget in the design is stated against: if this
    /// climbs past the arena size, the frame is spilling into a second chunk.
    #[must_use]
    pub const fn peak_vertices(&self) -> usize {
        self.batcher.peak()
    }

    pub(crate) fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub(crate) fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub(crate) fn queue_arc(&self) -> Arc<wgpu::Queue> {
        Arc::clone(&self.queue)
    }

    pub(crate) fn bind_texture(&self, texture: &wgpu::Texture, filter: Filter) -> wgpu::BindGroup {
        Self::bind(
            &self.device,
            &self.layout,
            texture,
            filter,
            "fathom.user_texture",
        )
    }

    pub(crate) fn parts(&mut self) -> (&wgpu::Device, &wgpu::Queue, &mut Batcher) {
        (&self.device, &self.queue, &mut self.batcher)
    }

    pub(crate) fn clear_color(&self) -> wgpu::Color {
        let [r, g, b, a] = self.clear.channels();
        let f = |v: u8| f64::from(v) / 255.0;
        wgpu::Color {
            r: f(r),
            g: f(g),
            b: f(b),
            a: f(a),
        }
    }

    /// Replay the frame's command list into one render pass and one submit.
    pub(crate) fn submit(&mut self, target: &Surfaced) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("fathom.frame"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("fathom.pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target.view(),
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.clear_color()),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            self.batcher.render(
                &self.device,
                &self.queue,
                &mut pass,
                &self.pipelines,
                &self.atlas,
            );
        }
        self.queue.submit(Some(encoder.finish()));
    }

    pub(crate) fn acquire(&self) -> Option<Surfaced> {
        match &self.output {
            Output::Surface { surface, .. } => {
                let frame = surface.get_current_texture().ok()?;
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                Some(Surfaced::Window(Box::new(frame), view))
            }
            Output::Offscreen { view, .. } => Some(Surfaced::Offscreen(view.clone())),
        }
    }

    /// Read the offscreen target back as RGBA8 rows, top row first.
    ///
    /// Returns `None` for a windowed context: there is nothing to read back
    /// that the compositor has not already taken.
    ///
    /// # Panics
    ///
    /// Never. The internal map callback cannot fail without the device being
    /// lost, in which case `poll` returns first and the result is `None`.
    #[must_use]
    pub fn read_pixels(&self) -> Option<Vec<u8>> {
        let Output::Offscreen { texture, .. } = &self.output else {
            return None;
        };
        let (width, height) = self.size;
        // Copies out of a texture want rows padded to 256 bytes.
        let unpadded = width * 4;
        let padded = unpadded.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fathom.readback"),
            size: u64::from(padded) * u64::from(height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("fathom.readback"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::PollType::Wait).ok()?;

        let mapped = slice.get_mapped_range();
        let mut out = Vec::with_capacity((unpadded * height) as usize);
        for row in mapped.chunks_exact(padded as usize) {
            out.extend_from_slice(row.get(..unpadded as usize)?);
        }
        drop(mapped);
        staging.unmap();
        Some(out)
    }
}

/// The acquired render target for one frame.
#[derive(Debug)]
pub(crate) enum Surfaced {
    Window(Box<wgpu::SurfaceTexture>, wgpu::TextureView),
    Offscreen(wgpu::TextureView),
}

impl Surfaced {
    pub(crate) fn view(&self) -> &wgpu::TextureView {
        match self {
            Self::Window(_, view) | Self::Offscreen(view) => view,
        }
    }

    pub(crate) fn present(self) {
        if let Self::Window(frame, _) = self {
            frame.present();
        }
    }
}
