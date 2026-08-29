# Live streaming and adapters

Live and playback are the same code path, because the library has no concept of
a "source". You pull a frame from wherever it came from and call
`update_texture`. That is the entire integration surface.

## Three rules

1. **No internal threads, no internal buffering.** The adapter crate owns the
   decode thread and the bounded channel. `update_texture` is a non-blocking
   staging write; if no new frame arrived, you simply do not call it and the
   previous texture is redrawn. A slow producer degrades to a stale frame, never
   to a stalled render loop.
2. **Timestamps, not frame indices, are the shared axis.** `Timestamp` is
   nanoseconds. Video arrives at 30Hz, joint states at 500Hz, a language
   instruction once per episode; a frame index cannot express that. Resampling
   is caller code and it is one line:

   ```rust
   # use fathom::Timestamp;
   # let states = [Timestamp(0), Timestamp(2_000_000)];
   # let now = Timestamp(3_000_000);
   let i = states.partition_point(|t| *t <= now) - 1;
   # assert_eq!(i, 1);
   ```

3. **Seeking does not exist in the library.** Scrubbing is you choosing which
   slice to hand to the draw calls. In live mode there is nothing to seek; in
   playback the adapter seeks. Same draw code either way.

## The adapter contract

MCAP, LeRobot, ROS bags, raw dumps and whatever comes next are **separate crates
outside this workspace**. The contract is deliberately thin: an adapter depends
on `fathom-core` alone and exposes plain slices of its types. No `Source` trait,
no plugin registry, no dynamic dispatch. An adapter is just a crate whose
functions happen to return types fathom already understands.

`fathom-core` is two dependencies deep - `glam` and `bytemuck` - and it stays
that way, because it is the only thing the ecosystem agrees on and every
breaking change to it breaks every adapter.

```rust,ignore
// what an adapter hands over: timestamped slices of core types, nothing more
pub struct Episode {
    pub instruction: String,
    pub intrinsics: Intrinsics,
    pub video: Vec<(Timestamp, Vec<u8>)>,       // RGBA8, ready for upload_texture
    pub joints: Vec<(Timestamp, [f32; 7])>,
    pub ee_path: Vec<(Timestamp, WorldPoint)>,  // already in world coordinates
}
```

The `adapter_stub` example is this contract as runnable code.

## Decode leaves the library

`fathom-media` keeps encode only, because "render N frames to an mp4" is an
output of the renderer. Decode belongs to the adapter that already knows the
container layout.
