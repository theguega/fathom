"""End-to-end checks for the Python bindings.

Run against a built extension:

    maturin develop -m crates/fathom-py/Cargo.toml
    python -m pytest crates/fathom-py/tests

It also runs as a plain script, so a contributor without pytest can check a
build with `python test_bindings.py`.
"""

import numpy as np
import pytest

import fathom

W2C = np.eye(4, dtype=np.float32)
K = (600.0, 600.0, 320.0, 240.0)


def test_colormap_channel_order():
    c = fathom.colormap(np.array([0.0, 0.5, 1.0], np.float32), 0.0, 1.0, "viridis")
    assert c.shape == (3, 4)
    assert c.dtype == np.uint8
    # Viridis starts at a dark purple. Red must not be the large channel here:
    # that was the symptom when Color packed into a u32 and came out reversed.
    assert c[0][0] < 100 and c[0][2] > 60, c[0]
    assert c[2][0] > 200 and c[2][1] > 200, c[2]


def test_colormap_rejects_an_unknown_ramp():
    with pytest.raises(ValueError, match="unknown colormap"):
        fathom.colormap(np.array([0.0], np.float32), 0.0, 1.0, "nope")


def test_project_and_unproject_round_trip():
    pts = np.array([[0, 0, 1], [0.3, -0.2, 2.0], [0, 0, -1]], np.float32)
    px = fathom.project(pts, K, W2C)

    assert px.shape == (3, 2)
    assert px[0] == pytest.approx([320.0, 240.0])
    # A point behind the camera has no pixel: a geometric outcome, not an error.
    assert np.isnan(px[2]).all()

    back = fathom.unproject(px[:2], np.array([1.0, 2.0], np.float32), K, W2C)
    assert back == pytest.approx(pts[:2], abs=1e-4)


def test_homography_round_trips():
    plane = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    image = np.array([[100, 100], [500, 120], [470, 400], [130, 380]], np.float32)

    h = fathom.homography_from_correspondences(plane, image)
    assert fathom.warp(plane, h) == pytest.approx(image, abs=1e-2)
    assert fathom.unwarp(image, h) == pytest.approx(plane, abs=1e-4)


def test_degenerate_correspondences_raise():
    collinear = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], np.float32)
    with pytest.raises(ValueError, match="singular"):
        fathom.homography_from_correspondences(collinear, collinear)


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(lambda a: np.asfortranarray(a), id="fortran_order"),
        pytest.param(lambda a: a[::2], id="strided_view"),
    ],
)
def test_non_contiguous_input_is_rejected_not_silently_reinterpreted(bad):
    pts = np.array([[0, 0, 1], [0.3, -0.2, 2.0], [0, 0, -1]], np.float32)
    with pytest.raises(ValueError, match="C-contiguous"):
        fathom.project(bad(pts), K, W2C)


def test_clear_colour_reaches_the_framebuffer():
    r = fathom.Renderer.headless(128, 96)
    r.set_clear_color((10, 20, 30, 255))
    r.begin_frame()
    r.end_frame()

    px = np.frombuffer(bytes(r.read_pixels()), np.uint8).reshape(96, 128, 4)
    assert tuple(px[0][0]) == (10, 20, 30, 255)


def test_the_frame_protocol_is_enforced_at_runtime():
    r = fathom.Renderer.headless(64, 64)

    with pytest.raises(RuntimeError, match="no frame is open"):
        fathom.draw_line_3d(r, (0, 0, 0), (1, 1, 1), (255, 0, 0, 255))

    r.begin_frame()
    with pytest.raises(RuntimeError, match="already open"):
        r.begin_frame()
    with pytest.raises(RuntimeError, match="no camera is bound"):
        fathom.draw_line_3d(r, (0, 0, 0), (1, 1, 1), (255, 0, 0, 255))
    r.end_frame()

    with pytest.raises(RuntimeError, match="no frame is open"):
        r.end_frame()


def test_a_real_frame_draws_2d_and_3d():
    r = fathom.Renderer.headless(128, 96)
    r.set_clear_color((0, 0, 0, 255))

    tex = r.upload_texture(bytes([255, 0, 0, 255] * 64), 8, 8, nearest=True)
    assert (tex.width, tex.height) == (8, 8)

    path = np.array([[-0.5, 0.2, 0], [0.5, 0.2, 0]], np.float32)
    colors = np.array([[0, 255, 0, 255], [0, 255, 255, 255]], np.uint8)

    r.begin_frame()
    fathom.draw_texture(r, tex, (0.0, 0.0, 32.0, 32.0))
    fathom.draw_text_at(r, (2.0, 40.0), "hello", 1)

    r.scene(fathom.Orbit(2.0).camera(128 / 96))
    fathom.draw_grid(r, 8, 0.25, (80, 80, 90, 255))
    fathom.draw_line_strip_3d_vc(r, path, colors)
    fathom.draw_points_3d(r, path, colors, 0.05)
    fathom.draw_frames(r, np.stack([np.eye(4, dtype=np.float32)] * 3), 0.2)
    r.end_scene()
    r.end_frame()

    img = np.frombuffer(bytes(r.read_pixels()), np.uint8).reshape(96, 128, 4)
    # The texture is pure red, drawn top-left. This is also the channel-order
    # check: a reversed Color would put blue here.
    assert tuple(img[4][4][:3]) == (255, 0, 0)
    assert int((img[:, :, :3].max(axis=2) > 40).sum()) > 200
    assert r.peak_vertices > 0


def test_mismatched_lengths_are_caught_at_the_boundary():
    r = fathom.Renderer.headless(64, 64)
    path = np.array([[0, 0, 0], [1, 0, 0]], np.float32)
    colors = np.array([[0, 255, 0, 255]], np.uint8)

    r.begin_frame()
    r.scene(fathom.Orbit(2.0).camera(1.0))
    with pytest.raises(ValueError, match="same length"):
        fathom.draw_points_3d(r, path, colors, 0.05)
    r.end_frame()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
