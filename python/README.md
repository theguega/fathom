# fathom for Python

The Rust free functions, mirrored. numpy arrays cross zero-copy, and you still
own the loop.

## Build and install

fathom is not on PyPI yet, so build the extension from this repo. With
[`uv`](https://docs.astral.sh/uv/):

```sh
uv venv --python 3.11 .venv
uv pip install --python .venv maturin numpy
VIRTUAL_ENV="$PWD/.venv" .venv/bin/maturin develop --release -m crates/fathom-py/Cargo.toml
```

Or with a plain venv:

```sh
python3.11 -m venv .venv && source .venv/bin/activate
pip install maturin numpy
maturin develop --release -m crates/fathom-py/Cargo.toml
```

**Use `--release`.** A debug build of the extension is roughly 24x slower, which
is enough to make any timing you take from it meaningless.

Python 3.10 or newer. One `abi3` wheel covers every version from there up.

## Run the examples

```sh
.venv/bin/python python/examples/scrub.py       # live viewer: video, 3D path, plot
.venv/bin/python python/examples/attention.py   # attention map blended over the camera
.venv/bin/python python/examples/outliers.py    # 20 demos coloured by distance
```

Drag to orbit, wheel to zoom, close the window to exit.

## Tests and benchmarks

```sh
uv pip install --python .venv pytest
.venv/bin/python -m pytest crates/fathom-py/tests -q

uv pip install --python .venv matplotlib plotly
.venv/bin/python python/benches/compare.py      # the table in the root README
```

## The shape of the API

You drive the loop; `poll()` pumps window events and returns `False` once the
window closes. Nothing calls back into your code.

```python
import numpy as np, fathom

r = fathom.Renderer.window("fathom", 1280, 720)
orbit = fathom.Orbit(2.0)

while r.poll():
    r.begin_frame()

    # Layout is Rect maths in your code; there is no panel manager.
    x, y, w, h = r.viewport
    fathom.draw_text_at(r, (16, 16), "episode 41", 2)

    # 3D needs a bound camera. Pass a panel to confine it to one rectangle.
    r.scene(orbit.camera((w / 2) / h), (w / 2, 0, w / 2, h))
    fathom.draw_grid(r, 20, 0.1, (45, 45, 55, 255))
    fathom.draw_line_strip_3d_vc(r, path, fathom.colormap(scores, 0, 1, "viridis"))
    r.end_scene()

    r.end_frame()
```

`Renderer.headless(w, h)` is the same API with no window, and `read_pixels()`
gets the frame back as RGBA8 — that is how the tests and benchmarks run.

## Zero-copy, and what it refuses

Arrays are borrowed, never copied: `(N, 3) float32` positions, `(N, 4) uint8`
colours, `(N, 4, 4) float32` transforms. Anything not C-contiguous raises rather
than silently reinterpreting — **including Fortran-ordered arrays**, which
`as_slice` alone would accept and read transposed:

```python
fathom.project(np.asfortranarray(pts), k, w2c)
# ValueError: array must be C-contiguous; call np.ascontiguousarray(a) first
```

Type stubs ship in `crates/fathom-py/fathom.pyi`, so editors and mypy work.
