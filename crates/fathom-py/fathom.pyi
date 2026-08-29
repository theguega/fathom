"""Type stubs for fathom.

Checked into the repo and diffed in CI, so editors and mypy work without a
build step. The docstrings live in the Rust source and are the single source of
truth; these signatures mirror them.
"""

from typing import Literal

import numpy as np
from numpy.typing import NDArray

__version__: str

Color = tuple[int, int, int, int]
"""(r, g, b, a), each 0-255."""

Rect = tuple[float, float, float, float]
"""(x, y, w, h) in pixels, origin at the top-left."""

Vec3 = tuple[float, float, float]
ColorMapName = Literal["turbo", "viridis", "magma", "grey", "gray", "coolwarm"]

class Texture:
    """A GPU texture. Clones address the same allocation."""

    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...
    @property
    def aspect(self) -> float: ...
    def update(self, data: bytes) -> None:
        """Replace the pixels from an RGBA8 buffer. Non-blocking."""

class Camera:
    """A bound viewpoint: view and projection, already multiplied."""

    def __init__(
        self,
        eye: Vec3,
        target: Vec3,
        up: Vec3 = (0.0, 1.0, 0.0),
        fov_y: float = 0.785,
        aspect: float = 1.777,
        near: float = 0.01,
        far: float = 100.0,
    ) -> None: ...

class Orbit:
    """A turntable viewpoint: plain state you drive from your own input."""

    def __init__(self, distance: float = 2.0) -> None: ...
    @property
    def distance(self) -> float: ...
    def rotate(self, dyaw: float, dpitch: float) -> None: ...
    def zoom(self, notches: float) -> None: ...
    def pan(self, dx: float, dy: float) -> None: ...
    def camera(self, aspect: float) -> Camera: ...
    def set_target(self, target: Vec3) -> None: ...
    def set_up(self, up: Vec3) -> None: ...

class Renderer:
    """The renderer, and the frame currently in flight."""

    @staticmethod
    def window(title: str = "fathom", width: int = 1280, height: int = 720) -> Renderer: ...
    @staticmethod
    def headless(width: int = 1280, height: int = 720) -> Renderer: ...
    @property
    def size(self) -> tuple[int, int]: ...
    @property
    def aspect(self) -> float: ...
    @property
    def viewport(self) -> Rect: ...
    @property
    def peak_vertices(self) -> int: ...
    def poll(self) -> bool:
        """Pump window events. False once the window has been closed."""

    def set_clear_color(self, rgba: Color) -> None: ...
    def upload_texture(
        self, data: bytes, width: int, height: int, nearest: bool = False
    ) -> Texture: ...
    def begin_frame(self) -> None: ...
    def scene(self, camera: Camera, panel: Rect | None = None) -> None: ...
    def end_scene(self) -> None: ...
    def end_frame(self) -> None: ...
    def read_pixels(self) -> list[int] | None: ...

# --- 2D, image and screen space ----------------------------------------------

def draw_texture(
    r: Renderer, tex: Texture, dst: Rect, tint: Color = (255, 255, 255, 255)
) -> None: ...
def draw_line_2d(
    r: Renderer, a: tuple[float, float], b: tuple[float, float], color: Color
) -> None: ...
def draw_line_strip_2d(r: Renderer, pts: NDArray[np.float32], color: Color) -> None: ...
def draw_bbox(r: Renderer, rect: Rect, color: Color) -> None: ...
def draw_polygon(r: Renderer, pts: NDArray[np.float32], color: Color) -> None: ...
def draw_text_at(
    r: Renderer,
    pos: tuple[float, float],
    text: str,
    size: int = 1,
    color: Color = (255, 255, 255, 255),
) -> None: ...
def text_width(text: str, size: int = 1) -> float: ...

# --- 3D, world space, needs a bound camera -----------------------------------

def draw_line_3d(r: Renderer, a: Vec3, b: Vec3, color: Color) -> None: ...
def draw_line_strip_3d(r: Renderer, pts: NDArray[np.float32], color: Color) -> None: ...
def draw_line_strip_3d_vc(
    r: Renderer, pts: NDArray[np.float32], colors: NDArray[np.uint8]
) -> None: ...
def draw_points_3d(
    r: Renderer, pts: NDArray[np.float32], colors: NDArray[np.uint8], size: float = 0.01
) -> None: ...
def draw_grid(
    r: Renderer,
    slices: int = 20,
    spacing: float = 0.1,
    color: Color = (128, 128, 128, 255),
) -> None: ...
def draw_frames(
    r: Renderer, transforms: NDArray[np.float32], axis_len: float = 0.05
) -> None: ...
def draw_wire_ellipsoid(
    r: Renderer, center: Vec3, axes: NDArray[np.float32], color: Color
) -> None: ...

# --- pure maths, no GPU ------------------------------------------------------

def colormap(
    values: NDArray[np.float32],
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap: ColorMapName = "turbo",
) -> NDArray[np.uint8]: ...
def project(
    points: NDArray[np.float32],
    k: tuple[float, float, float, float],
    world_to_camera: NDArray[np.float32],
    distortion: tuple[float, float, float, float, float] | None = None,
) -> NDArray[np.float32]: ...
def unproject(
    pixels: NDArray[np.float32],
    depth: NDArray[np.float32],
    k: tuple[float, float, float, float],
    world_to_camera: NDArray[np.float32],
    distortion: tuple[float, float, float, float, float] | None = None,
) -> NDArray[np.float32]: ...
def warp(points: NDArray[np.float32], h: NDArray[np.float32]) -> NDArray[np.float32]: ...
def unwarp(pixels: NDArray[np.float32], h: NDArray[np.float32]) -> NDArray[np.float32]: ...
def homography_from_correspondences(
    src: NDArray[np.float32], dst: NDArray[np.float32]
) -> NDArray[np.float32]: ...
