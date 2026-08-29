"""Same job, three stacks. Reproduces the table in the README.

    pip install matplotlib plotly numpy
    maturin develop --release -m crates/fathom-py/Cargo.toml
    python python/benches/compare.py

Build fathom with --release. A debug build of the extension is roughly 24x
slower here, which is enough to make the comparison meaningless.

Steady-state cost of one updated frame.

The job: a 3D trajectory of N points plus a scrolling time series, redrawn
every frame at 1280x720 - the loop a policy debugger actually runs.

Every stack is given its fast path, not its naive one:
  * matplotlib redraws with set_data/set_offsets on an existing artist and an
    Agg canvas, not a fresh figure per frame.
  * plotly is measured on serialisation only, because that is the per-update
    cost a browser-based viewer cannot avoid; the browser's own render is on
    top of this and is not counted here.
  * fathom is measured twice: with the GPU readback a headless run needs, and
    without it, since a windowed app presents instead of reading back.
"""
import time
import numpy as np

W, H = 1280, 720
FRAMES = 30


def timeit(fn, frames=FRAMES):
    fn()  # warm
    t0 = time.perf_counter()
    for i in range(frames):
        fn(i)
    return (time.perf_counter() - t0) / frames * 1000.0


def make_path(n, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 12, n, dtype=np.float32)
    return np.stack([np.cos(t) * 0.4, t * 0.02, np.sin(t) * 0.4], -1).astype(np.float32) \
        + rng.normal(0, 0.004, (n, 3)).astype(np.float32)


# ---------------------------------------------------------------- matplotlib
def bench_matplotlib(n):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = make_path(n)
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_subplot(121, projection="3d")
    line, = ax.plot(path[:, 0], path[:, 1], path[:, 2], lw=0.5)
    ax2 = fig.add_subplot(122)
    series = np.sin(np.linspace(0, 20, 512, dtype=np.float32))
    line2, = ax2.plot(series)
    fig.canvas.draw()

    def frame(i=0):
        p = path + np.float32(i) * 0.0005
        line.set_data_3d(p[:, 0], p[:, 1], p[:, 2])
        line2.set_ydata(np.roll(series, i))
        fig.canvas.draw()
        fig.canvas.buffer_rgba()

    ms = timeit(frame)
    plt.close(fig)
    return ms


# -------------------------------------------------------------------- plotly
def bench_plotly(n):
    import plotly.graph_objects as go

    path = make_path(n)
    series = np.sin(np.linspace(0, 20, 512, dtype=np.float32))

    def frame(i=0):
        p = path + np.float32(i) * 0.0005
        fig = go.Figure(
            [
                go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="lines"),
                go.Scatter(y=np.roll(series, i)),
            ]
        )
        # What the browser must receive for this update.
        return fig.to_json()

    return timeit(frame)


# -------------------------------------------------------------------- fathom
def bench_fathom(n, readback):
    import fathom

    path = make_path(n)
    colors = np.tile(np.array([[0, 255, 128, 255]], np.uint8), (n, 1))
    series = np.sin(np.linspace(0, 20, 512, dtype=np.float32))
    plot_pts = np.stack(
        [np.linspace(660, 1260, 512, dtype=np.float32), np.zeros(512, np.float32)], -1
    )

    r = fathom.Renderer.headless(W, H)
    orbit = fathom.Orbit(2.0)

    def frame(i=0):
        p = np.ascontiguousarray(path + np.float32(i) * 0.0005)
        pts = plot_pts.copy()
        pts[:, 1] = 360 - np.roll(series, i) * 150
        r.begin_frame()
        fathom.draw_line_strip_2d(r, pts, (0, 255, 0, 255))
        r.scene(orbit.camera(W / H), (0.0, 0.0, W / 2, float(H)))
        fathom.draw_grid(r, 20, 0.1, (45, 45, 55, 255))
        fathom.draw_line_strip_3d_vc(r, p, colors)
        r.end_scene()
        r.end_frame()
        if readback:
            r.read_pixels()

    return timeit(frame)


if __name__ == "__main__":
    print(f"{'points':>9} | {'matplotlib':>11} | {'plotly json':>12} | {'fathom+rb':>10} | {'fathom':>8}")
    print("-" * 66)
    for n in (1_000, 10_000, 100_000):
        mpl = bench_matplotlib(n)
        ply = bench_plotly(n)
        fa_rb = bench_fathom(n, True)
        fa = bench_fathom(n, False)
        print(f"{n:>9,} | {mpl:>8.1f} ms | {ply:>9.1f} ms | {fa_rb:>7.2f} ms | {fa:>5.2f} ms")
