//! GPU textures: video frames, false-colored depth, similarity matrices.
//!
//! Allocation takes a [`Ctx`](crate::Ctx), never a [`Frame`](crate::Frame), so
//! creating a texture mid-frame is a compile error rather than a stall you
//! profile later. Per-frame updates go through [`update_texture`], which is a
//! non-blocking staging write.

use std::{fmt, sync::Arc};

use wgpu::util::DeviceExt as _;

use crate::Ctx;

/// Byte layout of pixel data handed to [`upload_texture`].
///
/// Single-channel data is deliberately absent: a depth or attention map becomes
/// RGBA through [`colormap_into`](fathom_geom::colormap_into) first, which is
/// the same one-line step that makes it *readable*.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Format {
    /// Four bytes per pixel, red first. What `colormap` produces.
    Rgba8,
    /// Four bytes per pixel, blue first. What most capture APIs produce.
    Bgra8,
}

impl Format {
    #[inline]
    const fn bytes_per_pixel(self) -> u32 {
        4
    }

    #[inline]
    const fn wgpu(self) -> wgpu::TextureFormat {
        match self {
            Self::Rgba8 => wgpu::TextureFormat::Rgba8UnormSrgb,
            Self::Bgra8 => wgpu::TextureFormat::Bgra8UnormSrgb,
            // `Format` is `#[non_exhaustive]`; nothing else exists yet.
        }
    }
}

/// How a texture is sampled when drawn at a size other than its own.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Filter {
    /// Smooth. The right default for camera frames.
    #[default]
    Linear,
    /// Blocky. The right choice for label maps and for pixel-peeping.
    Nearest,
}

/// Why a texture could not be created or updated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum TextureError {
    /// The pixel buffer does not match `width * height * bytes_per_pixel`.
    WrongSize {
        /// Bytes the dimensions and format imply.
        expected: usize,
        /// Bytes actually supplied.
        got: usize,
    },
}

impl fmt::Display for TextureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongSize { expected, got } => {
                write!(f, "expected {expected} bytes of pixel data, got {got}")
            }
        }
    }
}

impl std::error::Error for TextureError {}

/// A GPU texture, ready to draw.
///
/// Cloning is cheap: the handle is reference-counted and clones address the
/// same GPU allocation, so a caller can keep one per stream and hand copies
/// around without thinking about it.
#[derive(Clone, Debug)]
pub struct Texture {
    queue: Arc<wgpu::Queue>,
    texture: wgpu::Texture,
    pub(crate) bind_group: Arc<wgpu::BindGroup>,
    width: u32,
    height: u32,
    format: Format,
}

impl Texture {
    /// Width in pixels.
    #[inline]
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Height in pixels.
    #[inline]
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Width divided by height, for [`Rect::fit_aspect`](fathom_core::Rect::fit_aspect).
    #[inline]
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn aspect(&self) -> f32 {
        self.width as f32 / self.height as f32
    }

    /// Byte layout this texture was created with.
    #[inline]
    #[must_use]
    pub const fn format(&self) -> Format {
        self.format
    }

    fn expected_bytes(&self) -> usize {
        (self.width * self.height * self.format.bytes_per_pixel()) as usize
    }
}

/// Upload pixel data to a new GPU texture.
///
/// Takes a [`Ctx`], not a [`Frame`](crate::Frame): allocation happens between
/// frames, by construction. `NonZeroU32` dimensions delete the zero-size check.
///
/// ```no_run
/// use std::num::NonZeroU32;
/// use fathom_render::{Ctx, Filter, Format, upload_texture};
///
/// # fn demo(ctx: &Ctx) -> Result<(), Box<dyn std::error::Error>> {
/// let (w, h) = (NonZeroU32::new(640).ok_or("nonzero")?, NonZeroU32::new(480).ok_or("nonzero")?);
/// let pixels = vec![0u8; 640 * 480 * 4];
/// let tex = upload_texture(ctx, &pixels, w, h, Format::Rgba8, Filter::Linear)?;
/// assert_eq!(tex.width(), 640);
/// # Ok(()) }
/// ```
///
/// # Errors
///
/// [`TextureError::WrongSize`] if `data` is not exactly `w * h * 4` bytes.
pub fn upload_texture(
    ctx: &Ctx,
    data: &[u8],
    w: std::num::NonZeroU32,
    h: std::num::NonZeroU32,
    fmt: Format,
    filter: Filter,
) -> Result<Texture, TextureError> {
    let (width, height) = (w.get(), h.get());
    let expected = (width * height * fmt.bytes_per_pixel()) as usize;
    if data.len() != expected {
        return Err(TextureError::WrongSize {
            expected,
            got: data.len(),
        });
    }

    let texture = ctx.device().create_texture_with_data(
        ctx.queue(),
        &wgpu::TextureDescriptor {
            label: Some("fathom.texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: fmt.wgpu(),
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::LayerMajor,
        data,
    );

    Ok(Texture {
        queue: ctx.queue_arc(),
        bind_group: Arc::new(ctx.bind_texture(&texture, filter)),
        texture,
        width,
        height,
        format: fmt,
    })
}

/// Replace a texture's pixels in place, keeping the same GPU allocation.
///
/// This is the whole live-streaming integration surface. It is a non-blocking
/// staging write: if no new frame arrived, the caller simply does not call it
/// and the previous texture is redrawn, so a slow producer degrades to a stale
/// frame rather than a stalled render loop.
///
/// # Errors
///
/// [`TextureError::WrongSize`] if `data` is not exactly `width * height * 4` bytes.
pub fn update_texture(tex: &Texture, data: &[u8]) -> Result<(), TextureError> {
    let expected = tex.expected_bytes();
    if data.len() != expected {
        return Err(TextureError::WrongSize {
            expected,
            got: data.len(),
        });
    }
    tex.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &tex.texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(tex.width * tex.format.bytes_per_pixel()),
            rows_per_image: Some(tex.height),
        },
        wgpu::Extent3d {
            width: tex.width,
            height: tex.height,
            depth_or_array_layers: 1,
        },
    );
    Ok(())
}
