from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.geometry import (
    apply_homography,
    plane_induced_homography,
    project_points,
    rotation_matrix_from_euler,
    transform_points,
)
from src.io_utils import imread, imwrite, write_csv, write_json
from src.visualization import label_to_color, save_match_plot


LOGGER = logging.getLogger(__name__)


@dataclass
class RoomPlane:
    plane_id: int
    name: str
    origin: np.ndarray
    u_vec: np.ndarray
    v_vec: np.ndarray
    normal: np.ndarray
    offset: float       # d such that n^T X + d = 0
    texture: np.ndarray
    texture_source: str

    @property
    def corners(self) -> np.ndarray:
        return np.array(
            [
                self.origin,
                self.origin + self.u_vec,
                self.origin + self.u_vec + self.v_vec,
                self.origin + self.v_vec,
            ],
            dtype=np.float64,
        )

    def world_from_uv(self, uv: np.ndarray) -> np.ndarray:
        uv_arr = np.asarray(uv, dtype=np.float64)
        return (
            self.origin[None, :]
            + uv_arr[:, :1] * self.u_vec[None, :]
            + uv_arr[:, 1:2] * self.v_vec[None, :]
        )


@dataclass
class RenderResult:
    image: np.ndarray
    mask: np.ndarray
    depth: np.ndarray
    projected_quads: List[np.ndarray]


@dataclass
class SyntheticScene:
    scene_dir: Path
    difficulty: str
    gt_homographies: np.ndarray
    source_mask: np.ndarray
    target_mask: np.ndarray
    correspondences: Dict[str, np.ndarray]


def make_intrinsics(image_size: Tuple[int, int], focal_length: float = 300.0) -> np.ndarray:
    height, width = image_size
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    return np.array(
        [[focal_length, 0.0, cx], [0.0, focal_length, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _make_plane(
    plane_id: int,
    name: str,
    origin: Sequence[float],
    u_vec: Sequence[float],
    v_vec: Sequence[float],
    normal: Sequence[float],
    texture: np.ndarray,
    texture_source: str,
) -> RoomPlane:
    origin_arr = np.asarray(origin, dtype=np.float64)
    normal_arr = np.asarray(normal, dtype=np.float64)
    normal_arr = normal_arr / max(np.linalg.norm(normal_arr), 1e-12)
    offset = -float(normal_arr @ origin_arr)
    if abs(offset) < 1e-12:
        raise ValueError(f"plane {name} has near-zero offset")
    return RoomPlane(
        plane_id=plane_id,
        name=name,
        origin=origin_arr,
        u_vec=np.asarray(u_vec, dtype=np.float64),
        v_vec=np.asarray(v_vec, dtype=np.float64),
        normal=normal_arr,
        offset=offset,
        texture=texture,
        texture_source=texture_source,
    )


def _procedural_texture(kind: str, rng: np.random.Generator, size: Tuple[int, int] = (720, 720)) -> np.ndarray:
    height, width = size
    y, x = np.indices((height, width))
    if kind == "floor":
        base = 120 + 45 * (((x // 72) + (y // 72)) % 2)
        image = np.dstack([base * 0.78, base * 0.94, base * 1.04])
    elif "wall" in kind:
        brick = 132 + 32 * (((x // 96) + ((y // 48) % 2)) % 2)
        mortar = ((x % 96 < 4) | (y % 48 < 4)) * 70
        base = np.clip(brick - mortar, 50, 220)
        tint = np.array([1.08, 0.98, 0.88]) if kind == "front_wall" else np.array([0.88, 1.00, 1.08])
        image = base[..., None] * tint[None, None, :]
    else:
        base = 130 + 40 * np.sin(x / 37.0) + 35 * np.cos(y / 43.0)
        image = np.dstack([base * 0.8, base * 1.05, base * 0.9])
    noise = rng.normal(0, 10, size=(height, width, 3))
    image = np.clip(image + noise, 0, 255).astype(np.uint8)
    for _ in range(28):
        color = rng.integers(40, 230, size=3).tolist()
        p1 = tuple(rng.integers(0, min(width, height), size=2).tolist())
        p2 = tuple(rng.integers(0, min(width, height), size=2).tolist())
        cv2.line(image, p1, p2, color, int(rng.integers(1, 4)), cv2.LINE_AA)
    cv2.putText(
        image, kind[:12], (35, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX, 1.15, (30, 30, 30), 3, cv2.LINE_AA,
    )
    return image


def _find_hpatches_images(hpatches_dir: Path) -> List[Path]:
    root = Path(hpatches_dir)
    if not root.exists():
        return []
    candidates = sorted(root.glob("*/1.ppm"))
    if not candidates:
        candidates = sorted(root.glob("hpatches-sequences-release/*/1.ppm"))
    if not candidates:
        candidates = sorted(root.rglob("1.ppm"))
    return candidates


def _load_textures(
    plane_names: Sequence[str],
    hpatches_dir: Path,
    rng: np.random.Generator,
    texture_source: str,
) -> List[Tuple[np.ndarray, str]]:
    if texture_source not in {"auto", "hpatches", "procedural"}:
        raise ValueError("texture_source must be auto, hpatches, or procedural")
    textures: List[Tuple[np.ndarray, str]] = []
    hpatches_images = _find_hpatches_images(hpatches_dir) if texture_source in {"auto", "hpatches"} else []
    use_hpatches = bool(hpatches_images) and texture_source in {"auto", "hpatches"}
    if texture_source == "hpatches" and not use_hpatches:
        LOGGER.warning("HPatches textures not found at %s; falling back to procedural", hpatches_dir)
    if use_hpatches:
        replace = len(hpatches_images) < len(plane_names)
        indices = rng.choice(len(hpatches_images), size=len(plane_names), replace=replace)
        for idx in indices:
            path = hpatches_images[int(idx)]
            textures.append((imread(path, cv2.IMREAD_COLOR), str(path)))
        return textures
    for name in plane_names:
        textures.append((_procedural_texture(name, rng), f"procedural:{name}"))
    return textures


def _plane_names_for_difficulty(difficulty: str, medium_side: str = "left_wall") -> List[str]:
    if difficulty == "easy":
        return ["front_wall", "floor"]
    if difficulty == "medium":
        if medium_side not in {"left_wall", "right_wall"}:
            raise ValueError("medium_side must be left_wall or right_wall")
        return ["front_wall", medium_side, "floor"]
    if difficulty == "hard":
        return ["front_wall", "left_wall", "right_wall", "floor"]
    raise ValueError("difficulty must be easy, medium, or hard")


def make_room_planes(
    difficulty: str,
    hpatches_dir: Path,
    rng: np.random.Generator,
    texture_source: str = "auto",
    medium_side: str = "left_wall",
) -> List[RoomPlane]:
    """Build room planes that fully cover the FOV of a wide-angle camera.

    Camera at origin, looking +z (forward). With focal=300 and image 600x800:
    - FOV_H = 2*atan(400/300) ≈ 106°, FOV_V = 2*atan(300/300) = 90°
    - At z=3: image spans x ∈ [-4.0, +4.0], y ∈ [-3.0, +3.0]
    - Front wall with hw=5, hh=4 covers ALL image pixels at z=3.
    - Floor at y=1.5 overrides lower portion (where z_floor < z_wall=3).
    - Side walls at x=±sw_x override left/right portions.
    """
    names = _plane_names_for_difficulty(difficulty, medium_side=medium_side)
    texture_items = _load_textures(names, hpatches_dir, rng, texture_source)
    texture_by_name = dict(zip(names, texture_items))

    # Front wall: large enough to cover the full image as a background plane.
    # At z=3, focal=300, image 800px wide: need hw >= 3*400/300 = 4.0 for full H coverage.
    fw_z = 3.0
    fw_hw = 5.0   # larger than needed → extends beyond image edges
    fw_hh = 4.0

    # Side walls at x = ±sw_x override left/right portions.
    # At x=sw_x: visible when pixel_x > (cx - sw_x*f/z_wall) = 399.5 - sw_x*300/3 = 399.5 - 100*sw_x
    # For sw_x=1.8: right wall visible when pixel_x > 399.5 - 180 = 219.5 (left 27% of image shows left wall)
    # Wait, left wall at x=-1.8 is visible at pixel_x < 219.5 (left side of image)
    if difficulty == "easy":
        sw_x = 1.8  # not used
    elif difficulty == "medium":
        sw_x = 1.8   # one side wall: ~25% of image
    else:  # hard
        sw_x = 1.5   # both side walls: ~19% each, slightly wider coverage

    # Floor at y=fl_y: visible where z_floor = fl_y*f/(pixel_y-cy) < fw_z=3
    # i.e. pixel_y > cy + fl_y*f/fw_z = 299.5 + 1.5*300/3 = 449.5 (lower ~25% of image)
    fl_y = 1.5
    sw_extent_z = 5.0   # side walls extend from z=0.1 to z=5.1
    fl_extent_z = 5.0   # floor extends from z=0.1 to z=5.1

    # Near z for floor/side-walls: must keep all 4 corners within ~2x image dims
    # to avoid ill-conditioned perspective transforms in depth computation.
    # focal=300, hw=5: pixel_x of near corner = 300*5/z_near+399.5
    # Keep < 2400px: z_near >= 300*5/2000 = 0.75 → use 0.80
    fl_z_near = 0.80
    sw_z_near = 0.80
    fl_z_extent = 5.0  # floor extends from z_near to z_near + fl_z_extent
    sw_z_extent = 5.0

    specs: Dict[str, tuple] = {
        "front_wall": (
            [-fw_hw, -fw_hh, fw_z],
            [2.0 * fw_hw, 0.0, 0.0],
            [0.0, 2.0 * fw_hh, 0.0],
            [0.0, 0.0, -1.0],
        ),
        "floor": (
            [-fw_hw, fl_y, fl_z_near],
            [2.0 * fw_hw, 0.0, 0.0],
            [0.0, 0.0, fl_z_extent],
            [0.0, -1.0, 0.0],
        ),
        "left_wall": (
            [-sw_x, -fw_hh, sw_z_near],
            [0.0, 0.0, sw_z_extent],
            [0.0, 2.0 * fw_hh, 0.0],
            [1.0, 0.0, 0.0],
        ),
        "right_wall": (
            [sw_x, -fw_hh, sw_z_near],
            [0.0, 0.0, sw_z_extent],
            [0.0, 2.0 * fw_hh, 0.0],
            [-1.0, 0.0, 0.0],
        ),
    }

    planes: List[RoomPlane] = []
    for plane_id, name in enumerate(names):
        texture, source = texture_by_name[name]
        planes.append(_make_plane(plane_id, name, *specs[name], texture=texture, texture_source=source))
    return planes


def _signed_magnitude(
    rng: np.random.Generator, min_abs: float, max_abs: float, *, allow_zero: bool = False
) -> float:
    if max_abs <= 0.0:
        return 0.0
    if allow_zero and min_abs <= 0.0:
        magnitude = rng.uniform(0.0, max_abs)
    else:
        magnitude = rng.uniform(min_abs, max_abs)
    sign = -1.0 if rng.random() < 0.5 else 1.0
    return float(sign * magnitude)


def _sample_natural_6dof_motion(
    difficulty: str, rng: np.random.Generator
) -> Tuple[float, float, float, np.ndarray]:
    if difficulty == "easy":
        yaw = _signed_magnitude(rng, 2.0, 5.0)
        pitch = _signed_magnitude(rng, 1.0, 3.0)
        roll = _signed_magnitude(rng, 0.0, 1.0, allow_zero=True)
        tx = _signed_magnitude(rng, 0.05, 0.12)
        ty = _signed_magnitude(rng, 0.0, 0.05, allow_zero=True)
        tz = _signed_magnitude(rng, 0.03, 0.10)
    elif difficulty == "medium":
        yaw = _signed_magnitude(rng, 5.0, 10.0)
        pitch = _signed_magnitude(rng, 2.0, 6.0)
        roll = _signed_magnitude(rng, 0.0, 2.0, allow_zero=True)
        tx = _signed_magnitude(rng, 0.10, 0.25)
        ty = _signed_magnitude(rng, 0.0, 0.08, allow_zero=True)
        tz = _signed_magnitude(rng, 0.08, 0.20)
    else:  # hard
        yaw = rng.uniform(-7.0, 7.0)
        pitch = _signed_magnitude(rng, 2.0, 5.0)
        roll = _signed_magnitude(rng, 0.0, 1.5, allow_zero=True)
        tx = _signed_magnitude(rng, 0.12, 0.25)
        ty = _signed_magnitude(rng, 0.0, 0.08, allow_zero=True)
        tz = _signed_magnitude(rng, 0.10, 0.22)
    t = np.array([tx, ty, tz], dtype=np.float64)
    return float(yaw), float(pitch), float(roll), t


def make_camera_motion(
    difficulty: str, rng: np.random.Generator, motion_mode: str = "natural_6dof"
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    if motion_mode != "natural_6dof":
        raise ValueError("CVproject9 supports only natural_6dof motion_mode")
    yaw, pitch, roll, t = _sample_natural_6dof_motion(difficulty, rng)
    rotation = rotation_matrix_from_euler(pitch_deg=pitch, yaw_deg=yaw, roll_deg=roll)
    motion_info = {
        "motion_mode": motion_mode,
        "yaw_deg": float(yaw),
        "pitch_deg": float(pitch),
        "roll_deg": float(roll),
        "tx": float(t[0]),
        "ty": float(t[1]),
        "tz": float(t[2]),
    }
    return rotation, t, motion_info


def _image_to_uv(projected_quad: np.ndarray) -> np.ndarray:
    uv_corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    return cv2.getPerspectiveTransform(projected_quad.astype(np.float32), uv_corners)


def _depth_for_pixels(
    plane: RoomPlane,
    projected_quad: np.ndarray,
    pixels_xy: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    h_img_to_uv = _image_to_uv(projected_quad)
    pts_h = np.column_stack([pixels_xy, np.ones(len(pixels_xy), dtype=np.float64)])
    uv_h = (h_img_to_uv @ pts_h.T).T
    uv = np.full((len(pixels_xy), 2), np.nan, dtype=np.float64)
    valid = np.abs(uv_h[:, 2]) > 1e-12
    uv[valid] = uv_h[valid, :2] / uv_h[valid, 2:3]
    world = plane.world_from_uv(uv)
    camera_points = transform_points(world, rotation, translation)
    return camera_points[:, 2]


def render_room(
    planes: Sequence[RoomPlane],
    intrinsics: np.ndarray,
    image_size: Tuple[int, int],
    rotation: Optional[np.ndarray] = None,
    translation: Optional[np.ndarray] = None,
) -> RenderResult:
    height, width = image_size
    r = np.eye(3, dtype=np.float64) if rotation is None else np.asarray(rotation, dtype=np.float64)
    t = np.zeros(3, dtype=np.float64) if translation is None else np.asarray(translation, dtype=np.float64).reshape(3)
    image = np.full((height, width, 3), 34, dtype=np.uint8)
    mask = np.full((height, width), -1, dtype=np.int32)
    depth_buffer = np.full((height, width), np.inf, dtype=np.float64)
    projected_quads: List[np.ndarray] = []

    for plane in planes:
        camera_corners = transform_points(plane.corners, r, t)
        quad, corner_depth = project_points(intrinsics, camera_corners)
        projected_quads.append(quad)
        if not np.all(np.isfinite(quad)) or not np.any(corner_depth > 1e-6):
            continue
        tex_h, tex_w = plane.texture.shape[:2]
        texture_corners = np.array(
            [[0, 0], [tex_w - 1, 0], [tex_w - 1, tex_h - 1], [0, tex_h - 1]], dtype=np.float32
        )
        h_tex_to_img = cv2.getPerspectiveTransform(texture_corners, quad.astype(np.float32))
        warped = cv2.warpPerspective(
            plane.texture, h_tex_to_img, (width, height),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
        )
        poly_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(poly_mask, np.round(quad).astype(np.int32), 1)
        ys, xs = np.nonzero(poly_mask)
        if len(xs) == 0:
            continue
        pixels = np.column_stack([xs, ys]).astype(np.float64)
        depths = _depth_for_pixels(plane, quad, pixels, r, t)
        update = np.isfinite(depths) & (depths > 1e-6) & (depths < depth_buffer[ys, xs])
        if np.any(update):
            yy = ys[update]
            xx = xs[update]
            image[yy, xx] = warped[yy, xx]
            mask[yy, xx] = plane.plane_id
            depth_buffer[yy, xx] = depths[update]
    return RenderResult(image=image, mask=mask, depth=depth_buffer, projected_quads=projected_quads)


def sample_correspondences(
    planes: Sequence[RoomPlane],
    intrinsics: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    samples_per_plane: int,
    rng: np.random.Generator,
    noise_std: float = 0.0,
) -> Dict[str, np.ndarray]:
    height, width = source_mask.shape
    src_all: List[np.ndarray] = []
    dst_all: List[np.ndarray] = []
    clean_src_all: List[np.ndarray] = []
    clean_dst_all: List[np.ndarray] = []
    label_all: List[np.ndarray] = []

    side = int(np.ceil(np.sqrt(samples_per_plane * 1.8)))
    grid = np.linspace(0.06, 0.94, side)
    uu, vv = np.meshgrid(grid, grid)
    base_uv_candidates = np.column_stack([uu.ravel(), vv.ravel()])

    for plane in planes:
        uv_candidates = base_uv_candidates
        source_quad, _ = project_points(intrinsics, plane.corners)
        if np.all(np.isfinite(source_quad)):
            ys, xs = np.nonzero(source_mask == plane.plane_id)
            if len(xs) > 0:
                pool_size = min(len(xs), max(samples_per_plane * 25, samples_per_plane))
                chosen = rng.choice(len(xs), size=pool_size, replace=False)
                pixels = np.column_stack([xs[chosen] + 0.5, ys[chosen] + 0.5]).astype(np.float64)
                h_img_to_uv = _image_to_uv(source_quad)
                uv_h = (h_img_to_uv @ np.column_stack([pixels, np.ones(len(pixels))]).T).T
                valid_uv = np.abs(uv_h[:, 2]) > 1e-12
                sampled_uv = np.full((len(pixels), 2), np.nan, dtype=np.float64)
                sampled_uv[valid_uv] = uv_h[valid_uv, :2] / uv_h[valid_uv, 2:3]
                inside = (
                    np.isfinite(sampled_uv).all(axis=1)
                    & np.all((sampled_uv >= 0.01) & (sampled_uv <= 0.99), axis=1)
                )
                if np.any(inside):
                    uv_candidates = np.vstack([uv_candidates, sampled_uv[inside]])
        world = plane.world_from_uv(uv_candidates)
        src_xy, src_depth = project_points(intrinsics, world)
        target_camera = transform_points(world, rotation, translation)
        dst_xy, dst_depth = project_points(intrinsics, target_camera)
        valid = (
            np.isfinite(src_xy).all(axis=1)
            & np.isfinite(dst_xy).all(axis=1)
            & (src_depth > 1e-6)
            & (dst_depth > 1e-6)
            & (src_xy[:, 0] >= 0)
            & (src_xy[:, 0] < width)
            & (src_xy[:, 1] >= 0)
            & (src_xy[:, 1] < height)
            & (dst_xy[:, 0] >= 0)
            & (dst_xy[:, 0] < width)
            & (dst_xy[:, 1] >= 0)
            & (dst_xy[:, 1] < height)
        )
        src_int = np.rint(src_xy).astype(np.int64)
        dst_int = np.rint(dst_xy).astype(np.int64)
        src_int[:, 0] = np.clip(src_int[:, 0], 0, width - 1)
        src_int[:, 1] = np.clip(src_int[:, 1], 0, height - 1)
        dst_int[:, 0] = np.clip(dst_int[:, 0], 0, width - 1)
        dst_int[:, 1] = np.clip(dst_int[:, 1], 0, height - 1)
        visible = valid.copy()
        visible[valid] &= source_mask[src_int[valid, 1], src_int[valid, 0]] == plane.plane_id
        visible[valid] &= target_mask[dst_int[valid, 1], dst_int[valid, 0]] == plane.plane_id
        indices = np.flatnonzero(visible)
        if len(indices) > samples_per_plane:
            indices = rng.choice(indices, size=samples_per_plane, replace=False)
        clean_src = src_xy[indices]
        clean_dst = dst_xy[indices]
        noisy_src = clean_src.copy()
        noisy_dst = clean_dst.copy()
        if noise_std > 0:
            noisy_src += rng.normal(0.0, noise_std, clean_src.shape)
            noisy_dst += rng.normal(0.0, noise_std, clean_dst.shape)
        src_all.append(noisy_src)
        dst_all.append(noisy_dst)
        clean_src_all.append(clean_src)
        clean_dst_all.append(clean_dst)
        label_all.append(np.full(len(indices), plane.plane_id, dtype=np.int32))

    if not src_all:
        return {
            "src_pts": np.empty((0, 2), dtype=np.float64),
            "dst_pts": np.empty((0, 2), dtype=np.float64),
            "clean_src_pts": np.empty((0, 2), dtype=np.float64),
            "clean_dst_pts": np.empty((0, 2), dtype=np.float64),
            "plane_labels": np.empty(0, dtype=np.int32),
        }
    return {
        "src_pts": np.vstack(src_all).astype(np.float64),
        "dst_pts": np.vstack(dst_all).astype(np.float64),
        "clean_src_pts": np.vstack(clean_src_all).astype(np.float64),
        "clean_dst_pts": np.vstack(clean_dst_all).astype(np.float64),
        "plane_labels": np.concatenate(label_all).astype(np.int32),
    }


def _save_preview(
    source: np.ndarray, target: np.ndarray,
    source_mask: np.ndarray, target_mask: np.ndarray, output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panels = [
        (source, "source"), (target, "target"),
        (label_to_color(source_mask), "source mask"),
        (label_to_color(target_mask), "target mask"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
    for ax, (image, title) in zip(axes.ravel(), panels):
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    r = np.asarray(rotation, dtype=np.float64)
    cos_angle = (np.trace(r) - 1.0) / 2.0
    return float(math.degrees(math.acos(float(np.clip(cos_angle, -1.0, 1.0)))))


def _plane_area_ratios(mask: np.ndarray, k_gt: int) -> Dict[str, float]:
    total = float(mask.size)
    return {str(k): float(np.sum(mask == k) / total) for k in range(k_gt)}


class SyntheticRoomGenerator:
    def __init__(
        self,
        config=None,
        output_dir: Optional[str | Path] = None,
        texture_source: str = "auto",
        seed: Optional[int] = None,
        motion_mode: Optional[str] = None,
    ) -> None:
        from config import Config, get_config
        self.config = config or get_config()
        self.output_dir = Path(output_dir) if output_dir is not None else Path(self.config.synthetic_dir)
        self.texture_source = texture_source
        self.seed = int(self.config.random_seed if seed is None else seed)
        self.motion_mode = str(motion_mode or getattr(self.config, "synthetic_motion_mode", "natural_6dof"))
        if self.motion_mode != "natural_6dof":
            raise ValueError("CVproject9 supports only natural_6dof motion_mode")

    def generate_scene(
        self,
        scene_index: int,
        difficulty: str,
        output_dir: Optional[str | Path] = None,
        noise_std: Optional[float] = None,
        seed_offset: int = 0,
    ) -> SyntheticScene:
        scene_id = f"scene_{scene_index:03d}"
        scene_dir = Path(output_dir) if output_dir is not None else self.output_dir / scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(
            self.seed
            + 1009 * scene_index
            + 104729 * seed_offset
            + {"easy": 0, "medium": 10000, "hard": 20000}[difficulty]
        )
        focal = float(getattr(self.config, "focal_length", 300.0))
        intrinsics = make_intrinsics(self.config.image_size, focal_length=focal)
        rotation, translation, motion_info = make_camera_motion(difficulty, rng, self.motion_mode)
        yaw_deg = float(motion_info["yaw_deg"])
        medium_side = "right_wall" if difficulty == "medium" and yaw_deg < 0 else "left_wall"
        planes = make_room_planes(
            difficulty,
            Path(self.config.hpatches_dir),
            rng,
            self.texture_source,
            medium_side=medium_side,
        )
        source = render_room(planes, intrinsics, self.config.image_size)
        target = render_room(planes, intrinsics, self.config.image_size, rotation, translation)
        gt_homographies = np.stack(
            [
                plane_induced_homography(intrinsics, rotation, translation, plane.normal, plane.offset)
                for plane in planes
            ],
            axis=0,
        )
        sigma = float(
            self.config.synthetic_noise_std.get(difficulty, 0.0) if noise_std is None else noise_std
        )
        correspondences = sample_correspondences(
            planes, intrinsics, rotation, translation,
            source.mask, target.mask,
            samples_per_plane=int(self.config.synthetic_samples_per_plane),
            rng=rng, noise_std=sigma,
        )

        imwrite(scene_dir / "source.png", source.image)
        imwrite(scene_dir / "target.png", target.image)
        imwrite(scene_dir / "source_mask.png", label_to_color(source.mask))
        imwrite(scene_dir / "target_mask.png", label_to_color(target.mask))
        np.save(scene_dir / "plane_mask_source.npy", source.mask)
        np.save(scene_dir / "plane_mask_target.npy", target.mask)
        np.save(scene_dir / "gt_homographies.npy", gt_homographies)
        np.savez(scene_dir / "correspondences.npz", **correspondences)
        _save_preview(source.image, target.image, source.mask, target.mask, scene_dir / "preview.png")
        save_match_plot(
            source.image, target.image,
            correspondences["src_pts"], correspondences["dst_pts"],
            correspondences["plane_labels"],
            scene_dir / "correspondences_preview.png",
        )

        invalid_source = float(np.mean(source.mask < 0))
        invalid_target = float(np.mean(target.mask < 0))

        max_h_error = 0.0
        if len(correspondences["clean_src_pts"]):
            for plane_id, h in enumerate(gt_homographies):
                idx = correspondences["plane_labels"] == plane_id
                if np.any(idx):
                    warped = apply_homography(h, correspondences["clean_src_pts"][idx])
                    error = np.linalg.norm(warped - correspondences["clean_dst_pts"][idx], axis=1)
                    max_h_error = max(max_h_error, float(np.nanmax(error)))

        metadata = {
            "scene_id": scene_id,
            "difficulty": difficulty,
            "homography_formula": "H = K (R - t n^T / d) K^{-1}",
            "K": intrinsics.tolist(),
            "R": rotation.tolist(),
            "t": translation.tolist(),
            "motion": motion_info,
            "invalid_source_fraction": invalid_source,
            "invalid_target_fraction": invalid_target,
            "max_clean_homography_error_px": max_h_error,
            "n_correspondences": int(len(correspondences["src_pts"])),
            "planes": [
                {
                    "plane_id": plane.plane_id,
                    "name": plane.name,
                    "normal": plane.normal.tolist(),
                    "offset": float(plane.offset),
                    "texture_source": plane.texture_source,
                    "source_quad": source.projected_quads[plane.plane_id].tolist(),
                    "target_quad": target.projected_quads[plane.plane_id].tolist(),
                }
                for plane in planes
            ],
        }
        write_json(scene_dir / "gt_camera_params.json", metadata)

        return SyntheticScene(
            scene_dir, difficulty, gt_homographies, source.mask, target.mask, correspondences
        )

    def quality_check(self, scene: SyntheticScene) -> Tuple[bool, List[str]]:
        """Returns (pass, reasons_for_rejection)."""
        reasons: List[str] = []
        invalid_source = float(np.mean(scene.source_mask < 0))
        invalid_target = float(np.mean(scene.target_mask < 0))
        max_ratio = float(getattr(self.config, "invalid_pixel_max_ratio", 0.10))
        if invalid_source >= max_ratio:
            reasons.append(f"invalid_source={invalid_source:.3f} >= {max_ratio}")
        if invalid_target >= max_ratio:
            reasons.append(f"invalid_target={invalid_target:.3f} >= {max_ratio}")

        k_gt = len(scene.gt_homographies)
        source_area = _plane_area_ratios(scene.source_mask, k_gt)
        target_area = _plane_area_ratios(scene.target_mask, k_gt)
        area_min = min(list(source_area.values()) + list(target_area.values())) if k_gt > 0 else 0.0
        if area_min < 0.08:
            reasons.append(f"min_plane_area={area_min:.3f} < 0.08")

        # Verify GT homography accuracy
        if len(scene.correspondences["clean_src_pts"]):
            all_errs = []
            for pid, h in enumerate(scene.gt_homographies):
                idx = scene.correspondences["plane_labels"] == pid
                if np.any(idx):
                    warped = apply_homography(h, scene.correspondences["clean_src_pts"][idx])
                    errs = np.linalg.norm(warped - scene.correspondences["clean_dst_pts"][idx], axis=1)
                    all_errs.extend(errs[np.isfinite(errs)].tolist())
            if all_errs and float(np.mean(all_errs)) > 0.5:
                reasons.append(f"gt_h_error_mean={np.mean(all_errs):.4f} > 0.5px")

        return len(reasons) == 0, reasons

    def generate_dataset(
        self,
        counts: Optional[Dict[str, int]] = None,
        max_attempts: int = 20,
        output_dir: Optional[str | Path] = None,
    ) -> List[SyntheticScene]:
        counts = counts or dict(self.config.synthetic_counts)
        out_dir = Path(output_dir) if output_dir is not None else self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        scenes: List[SyntheticScene] = []
        rows: List[Dict[str, object]] = []
        scene_index = 0
        for difficulty in ("easy", "medium", "hard"):
            for _ in range(int(counts.get(difficulty, 0))):
                scene_dir = out_dir / f"scene_{scene_index:03d}"
                accepted: Optional[SyntheticScene] = None
                for attempt in range(max_attempts):
                    scene = self.generate_scene(
                        scene_index, difficulty, output_dir=scene_dir,
                        noise_std=0.0, seed_offset=attempt,
                    )
                    ok, reasons = self.quality_check(scene)
                    if ok:
                        accepted = scene
                        LOGGER.info(
                            "Accepted %s (%s) attempt %d", scene_dir.name, difficulty, attempt + 1
                        )
                        break
                    LOGGER.info(
                        "Rejected %s (%s) attempt %d: %s",
                        scene_dir.name, difficulty, attempt + 1, "; ".join(reasons),
                    )
                if accepted is None:
                    LOGGER.warning(
                        "Keeping final failed candidate for %s after %d attempts",
                        scene_dir.name, max_attempts,
                    )
                    accepted = scene
                scenes.append(accepted)
                rows.append({
                    "scene_id": accepted.scene_dir.name,
                    "difficulty": difficulty,
                    "path": str(accepted.scene_dir),
                    "K_gt": len(accepted.gt_homographies),
                })
                scene_index += 1
        write_csv(out_dir / "scene_list.csv", rows)
        LOGGER.info("Generated %d synthetic room scenes in %s", len(scenes), out_dir)
        return scenes
