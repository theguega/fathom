//! `finish` consumes the open encoder and returns a different type, so writing
//! to a finished one is not a runtime error - it is not expressible.
use fathom::media::{EncodeOptions, Encoder};

fn demo() -> Result<(), fathom::media::MediaError> {
    let mut enc = Encoder::new("out.mp4", 16, 16, &EncodeOptions::default())?;
    enc.write(&[0; 1024])?;
    let mut done = enc.finish()?;
    done.write(&[0; 1024])?;
    Ok(())
}

fn main() {}
