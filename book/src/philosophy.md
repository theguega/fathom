# Philosophy

Borrowed wholesale from raylib. Not the dependency, the design discipline.

1. **Every primitive is a free function.** `draw_line_strip_3d(&mut s, &points, color)`.
   No builders, no objects with a lifecycle, no scene graph.
2. **Plain data in.** `&[WorldPoint]`, `&[f32]`, `Rect`. Never "load it into the
   system first" beyond the one unavoidable GPU upload.
3. **The caller owns all state.** Want a trajectory to persist across frames?
   Keep the `Vec` and call the draw function every frame. The library keeps no
   registry.
4. **The caller owns the loop.** `begin_frame` / draw / `end` is visible and
   driven by you. The library never calls back into your code.
5. **Composition is your code, not our abstraction.** Multi-panel layouts are
   `Rect` maths in your script. Analysis is your stats crate. We draw.
6. **The library ships no analysis.** Variance, similarity, outliers and
   clustering are *examples* that prove the primitive set is sufficient, not
   features.

## The design test

For any proposed addition, ask: does it need a new **noun** in the library, or
just a new **verb over slices**? Nouns are features. Verbs are primitives.
Reject nouns.

That test is why `colormap` exists and a heatmap primitive does not. A heatmap
is a texture, and a texture is already there.

## Build it like a 1990s game

Not nostalgia: a set of constraints that happen to produce the fastest, smallest
thing that does the job. A sprite-era engine had a fixed VRAM budget, a fixed
sprite count, one blitter, and a loop that had to finish inside a frame. Those
limits are what made them simple.

- **Allocate at init, never during a frame.** One vertex arena, sized at
  startup. Draw calls append into it. Overflow hands the full arena to another
  GPU buffer and carries on; it never reallocates and never drops anything.
- **Two pipelines, total.** Textured quads and lines. Every primitive lowers to
  one of them. Video is a quad, a depth heatmap is a quad, text is quads, a
  point cloud is quads, a trajectory is lines, an axis triad is lines. Adding a
  primitive means writing a lowering function, never a new shader path.
- **One texture atlas.** The font and the white texel that untextured
  primitives sample live in a single atlas, bound once.
- **State is regenerated every frame, never diffed.** Immediate mode is the
  whole point: there is no retained tree to invalidate, so there is no
  invalidation bug class.
- **Fixed budget, stated as a number.** Under 1ms of CPU to pack 100k vertices,
  one submit per frame. Criterion measures it; it currently runs at ~350µs.
- **Plain old data everywhere.** `Color` is `#[repr(transparent)]` over `u32`,
  `Vertex` is `#[repr(C)]` and `Pod`. Everything memcpys to the GPU through
  `bytemuck::cast_slice`, with no serialization step.

Two tests hold the line mechanically rather than by review: a counting global
allocator asserts zero heap allocations across a frame's draw calls, and
`trybuild` asserts that each type guarantee is still a compile error.
