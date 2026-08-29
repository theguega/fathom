# Port your matplotlib script

The mental shift is that there is no figure, no axes object and no `show()`.
There is a loop you own, and functions that draw into the current frame.

## A line plot

```python
# matplotlib
fig, ax = plt.subplots()
ax.plot(t, joint0, color="green")
ax.set_xlim(0, 10); ax.set_ylim(-1, 1)
plt.show()
```

```rust,no_run
# use fathom::prelude::*;
# fn demo(f: &mut Frame<'_>, joint0: &[f32], scratch: &mut Vec<ImagePoint>) {
// fathom: you own the axis scaling, because you own the Rect
let plot = f.viewport().inset(16.0);
draw_bbox(f, plot, Color::rgb(50, 50, 60));

scratch.clear();
scratch.extend(joint0.iter().enumerate().map(|(i, &v)| {
    let x = plot.x + plot.w * i as f32 / joint0.len() as f32;
    ImagePoint::new(x, plot.y + plot.h * 0.5 - v * plot.h * 0.45)
}));
draw_line_strip_2d(f, scratch, Color::GREEN);
# }
```

Longer, and that is the trade: no axis autoscaling, no ticks, no legend. In
exchange it runs at 60fps over a live stream, and the scaling is code you can
read rather than a `set_ylim` you have to guess at.

## `imshow` of a scalar field

```python
plt.imshow(depth, cmap="turbo", vmin=1.0, vmax=2.5)
```

```rust,no_run
# use std::num::NonZeroU32;
# use fathom::prelude::*;
# fn demo(ctx: &Ctx, f: &mut Frame<'_>, depth: &[f32], w: NonZeroU32, h: NonZeroU32) -> Result<(), Box<dyn std::error::Error>> {
let colors = colormap(depth, 1.0..2.5, ColorMap::Turbo);
let tex = upload_texture(ctx, bytemuck::cast_slice(&colors), w, h, Format::Rgba8, Filter::Nearest)?;
draw_texture(f, &tex, f.viewport().fit_aspect(tex.aspect()), Color::WHITE);
# Ok(()) }
```

`vmin`/`vmax` became the range argument, `cmap` became the `ColorMap`, and
`imshow` became "a heatmap is a texture". Upload once, outside the loop; call
`update_texture` per frame if the data changes.

## `scatter` with a colour dimension

```python
plt.scatter(xs, ys, c=scores, cmap="viridis")
```

```rust,no_run
# use fathom::prelude::*;
# fn demo(s: &mut Scene<'_, '_>, pts: &[WorldPoint], scores: &[f32], scratch: &mut Vec<(WorldPoint, Color)>) {
scratch.clear();
scratch.extend(pts.iter().zip(colormap(scores, 0.0..1.0, ColorMap::Viridis)).map(|(&p, c)| (p, c)));
draw_points_3d(s, scratch, Meters(0.01));
# }
```

The `c=` argument became an explicit `colormap` call. That is the pattern
throughout: anything matplotlib does implicitly, you do in one visible line.

## `subplot`

```python
fig, (ax1, ax2) = plt.subplots(1, 2)
```

```rust,no_run
# use fathom::prelude::*;
# fn demo(f: &mut Frame<'_>) {
let [left, right] = f.viewport().split_h();
# }
```

There is no panel manager. `Rect` has `split_h`, `split_v`, `inset` and
`fit_aspect`, and anything else is arithmetic.

## What you gain

The same draw code runs live, over a scrubbed recording, and into an mp4 with no
changes - see the `live_viewer` and `headless_export` examples, whose draw
sections are byte-identical.
