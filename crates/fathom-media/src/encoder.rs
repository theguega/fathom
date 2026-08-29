//! A typestate encoder: `Open` accepts frames, `Finished` does not exist until
//! the file is closed, so writing after finishing is unrepresentable.

use std::{
    fmt,
    io::Write as _,
    marker::PhantomData,
    path::Path,
    process::{Child, ChildStdin, Command, Stdio},
};

/// Why an encode could not start, continue, or complete.
#[derive(Debug)]
#[non_exhaustive]
pub enum MediaError {
    /// `ffmpeg` is not on `PATH`, or could not be started.
    NotInstalled(std::io::Error),
    /// A frame was not exactly `width * height * 4` bytes.
    WrongFrameSize {
        /// Bytes the dimensions imply.
        expected: usize,
        /// Bytes actually supplied.
        got: usize,
    },
    /// The encoder closed its input early, usually because it rejected the
    /// arguments. Its own diagnostics went to stderr.
    Closed(std::io::Error),
    /// `ffmpeg` exited with a failure status.
    Failed(Option<i32>),
}

impl fmt::Display for MediaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotInstalled(_) => {
                f.write_str("could not start ffmpeg; is it installed and on PATH?")
            }
            Self::WrongFrameSize { expected, got } => {
                write!(f, "expected {expected} bytes per frame, got {got}")
            }
            Self::Closed(_) => f.write_str("ffmpeg closed its input before the last frame"),
            Self::Failed(Some(code)) => write!(f, "ffmpeg exited with status {code}"),
            Self::Failed(None) => f.write_str("ffmpeg was killed by a signal"),
        }
    }
}

impl std::error::Error for MediaError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::NotInstalled(e) | Self::Closed(e) => Some(e),
            _ => None,
        }
    }
}

/// How to encode. The defaults are a sane H.264 that plays everywhere.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct EncodeOptions {
    /// Output frame rate.
    pub fps: u32,
    /// Constant rate factor: lower is better quality and a larger file.
    /// 18 is visually lossless, 23 is the ffmpeg default, 28 is small.
    pub crf: u8,
    /// x264 preset, trading encode speed against file size.
    pub preset: &'static str,
    /// Video codec, as ffmpeg names it.
    pub codec: &'static str,
}

impl EncodeOptions {
    /// The defaults at a chosen frame rate.
    ///
    /// The struct is `#[non_exhaustive]` so that new knobs are not a breaking
    /// change, which also means downstream cannot use struct-update syntax:
    /// start here and assign the public fields you care about.
    ///
    /// ```
    /// use fathom_media::EncodeOptions;
    ///
    /// let mut options = EncodeOptions::new(60);
    /// options.crf = 23;
    /// assert_eq!(options.fps, 60);
    /// ```
    #[must_use]
    pub fn new(fps: u32) -> Self {
        Self {
            fps,
            ..Self::default()
        }
    }
}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self {
            fps: 30,
            crf: 18,
            preset: "medium",
            codec: "libx264",
        }
    }
}

/// The encoder is accepting frames.
#[derive(Debug)]
pub struct Open;

/// The file is closed and complete.
#[derive(Debug)]
pub struct Finished;

/// An in-progress or completed encode.
///
/// The state parameter is the protocol: [`Encoder::write`] exists only on
/// `Encoder<Open>`, and [`Encoder::finish`] consumes it, so writing to a
/// finished encoder does not compile.
#[derive(Debug)]
pub struct Encoder<S> {
    child: Option<Child>,
    stdin: Option<ChildStdin>,
    frame_bytes: usize,
    frames: u64,
    _state: PhantomData<fn() -> S>,
}

impl Encoder<Open> {
    /// Start encoding to `path`, expecting `width` by `height` RGBA8 frames.
    ///
    /// # Errors
    ///
    /// [`MediaError::NotInstalled`] if `ffmpeg` cannot be started.
    pub fn new(
        path: impl AsRef<Path>,
        width: u32,
        height: u32,
        options: &EncodeOptions,
    ) -> Result<Self, MediaError> {
        let (width, height) = (width.max(1), height.max(1));
        let mut child = Command::new("ffmpeg")
            .args(["-hide_banner", "-loglevel", "error", "-y"])
            // Input: exactly what `Ctx::read_pixels` hands back.
            .args(["-f", "rawvideo", "-pix_fmt", "rgba"])
            .args(["-s", &format!("{width}x{height}")])
            .args(["-r", &options.fps.to_string()])
            .args(["-i", "-"])
            // Output: yuv420p so that every player will take it.
            .args(["-c:v", options.codec])
            .args(["-preset", options.preset])
            .args(["-crf", &options.crf.to_string()])
            .args(["-pix_fmt", "yuv420p"])
            .arg(path.as_ref())
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .spawn()
            .map_err(MediaError::NotInstalled)?;

        let stdin = child.stdin.take();
        Ok(Self {
            child: Some(child),
            stdin,
            frame_bytes: (width as usize) * (height as usize) * 4,
            frames: 0,
            _state: PhantomData,
        })
    }

    /// Append one RGBA8 frame, top row first.
    ///
    /// # Errors
    ///
    /// [`MediaError::WrongFrameSize`] if the buffer does not match the
    /// dimensions given to [`Encoder::new`], or [`MediaError::Closed`] if the
    /// encoder went away.
    pub fn write(&mut self, rgba: &[u8]) -> Result<(), MediaError> {
        if rgba.len() != self.frame_bytes {
            return Err(MediaError::WrongFrameSize {
                expected: self.frame_bytes,
                got: rgba.len(),
            });
        }
        let stdin = self.stdin.as_mut().ok_or_else(|| {
            MediaError::Closed(std::io::Error::from(std::io::ErrorKind::BrokenPipe))
        })?;
        stdin.write_all(rgba).map_err(MediaError::Closed)?;
        self.frames += 1;
        Ok(())
    }

    /// Close the input, wait for the file to be written, and consume `self`.
    ///
    /// # Errors
    ///
    /// [`MediaError::Failed`] if ffmpeg exited non-zero; its diagnostics will
    /// have gone to stderr.
    pub fn finish(mut self) -> Result<Encoder<Finished>, MediaError> {
        // Dropping stdin is what tells ffmpeg the stream is over.
        drop(self.stdin.take());
        let frames = self.frames;
        if let Some(mut child) = self.child.take() {
            let status = child.wait().map_err(MediaError::Closed)?;
            if !status.success() {
                return Err(MediaError::Failed(status.code()));
            }
        }
        Ok(Encoder {
            child: None,
            stdin: None,
            frame_bytes: self.frame_bytes,
            frames,
            _state: PhantomData,
        })
    }
}

impl Encoder<Finished> {
    /// How many frames were written.
    #[must_use]
    pub const fn frames(&self) -> u64 {
        self.frames
    }
}

impl<S> Drop for Encoder<S> {
    /// An encoder dropped without [`Encoder::finish`] leaves a truncated file
    /// rather than a hung process.
    fn drop(&mut self) {
        drop(self.stdin.take());
        if let Some(child) = &mut self.child {
            let _ = child.wait();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_frame_of_the_wrong_size_is_rejected_before_it_reaches_the_encoder() {
        let Ok(mut enc) = Encoder::new(
            std::env::temp_dir().join("fathom-size-check.mp4"),
            16,
            16,
            &EncodeOptions::default(),
        ) else {
            eprintln!("skipping: ffmpeg not installed");
            return;
        };
        assert!(matches!(
            enc.write(&[0; 10]),
            Err(MediaError::WrongFrameSize {
                expected: 1024,
                got: 10
            })
        ));
    }

    #[test]
    fn encodes_a_real_file() {
        let path = std::env::temp_dir().join("fathom-encode-test.mp4");
        let _ = std::fs::remove_file(&path);

        let mut options = EncodeOptions::new(10);
        options.preset = "ultrafast";
        let Ok(mut enc) = Encoder::new(&path, 64, 64, &options) else {
            eprintln!("skipping: ffmpeg not installed");
            return;
        };

        for i in 0..10u8 {
            let mut frame = vec![0u8; 64 * 64 * 4];
            for texel in frame.chunks_exact_mut(4) {
                texel.copy_from_slice(&[i * 25, 40, 200 - i * 15, 255]);
            }
            enc.write(&frame).unwrap();
        }

        let done = enc.finish().unwrap();
        assert_eq!(done.frames(), 10);

        let size = std::fs::metadata(&path).map_or(0, |m| m.len());
        assert!(
            size > 200,
            "the mp4 should have real content, got {size} bytes"
        );
        let _ = std::fs::remove_file(&path);
    }
}
