"""Background-aware dominant colour extraction in perceptual LAB space."""

from __future__ import annotations

from collections import deque

import numpy as np
from PIL import Image


PALETTE = {
    "black": "#181818",
    "white": "#f4f2ed",
    "grey": "#787878",
    "cream": "#eadfbd",
    "beige": "#c8ad8d",
    "brown": "#68462f",
    "navy": "#17254f",
    "blue": "#3978ac",
    "teal": "#187b80",
    "green": "#4f7a45",
    "red": "#c5373f",
    "burgundy": "#722f3a",
    "pink": "#d986a3",
    "purple": "#79558f",
    "yellow": "#d2aa32",
    "orange": "#c46b32",
}


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB values from 0..255 to CIE L*a*b* (D65)."""
    values = np.asarray(rgb, dtype=np.float32) / 255.0
    linear = np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)
    xyz = linear @ np.array(
        [[0.4124564, 0.2126729, 0.0193339],
         [0.3575761, 0.7151522, 0.1191920],
         [0.1804375, 0.0721750, 0.9503041]],
        dtype=np.float32,
    )
    xyz /= np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    transformed = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack(
        [116 * transformed[..., 1] - 16,
         500 * (transformed[..., 0] - transformed[..., 1]),
         200 * (transformed[..., 1] - transformed[..., 2])],
        axis=-1,
    )


def _hex_rgb(value: str) -> np.ndarray:
    return np.array([int(value[index:index + 2], 16) for index in (1, 3, 5)], dtype=np.float32)


PALETTE_NAMES = tuple(PALETTE)
PALETTE_LAB = _rgb_to_lab(np.stack([_hex_rgb(value) for value in PALETTE.values()]))
CALIBRATION_MAX_DISTANCE = 3.0


def _palette_name(centre: np.ndarray) -> str:
    lightness, green_red, blue_yellow = centre
    chroma = float(np.hypot(green_red, blue_yellow))
    if green_red < -15 and blue_yellow > 5:
        return "green"
    if 6 <= chroma < 25 and green_red > 2 and blue_yellow > 3:
        if lightness > 82:
            return "cream"
        return "beige" if lightness > 55 else "brown"
    if chroma < 9:
        if lightness < 28:
            return "black"
        if lightness > 88:
            return "white"
        return "grey"
    if blue_yellow < -8 and chroma > 10:
        if green_red < -7:
            return "teal"
        return "navy" if lightness < 32 else "blue"
    palette_index = int(np.argmin(np.linalg.norm(PALETTE_LAB - centre, axis=1)))
    return PALETTE_NAMES[palette_index]


def _garment_pixels(image: Image.Image) -> np.ndarray:
    """Remove an edge-connected, nearly uniform catalogue-photo background."""
    prepared = image.convert("RGB")
    prepared.thumbnail((160, 160), Image.Resampling.LANCZOS)
    rgb = np.asarray(prepared, dtype=np.float32)
    lab = _rgb_to_lab(rgb)
    border = np.concatenate((lab[0], lab[-1], lab[:, 0], lab[:, -1]))
    background = np.median(border, axis=0)
    border_distance = np.linalg.norm(border - background, axis=1)
    threshold = float(np.clip(np.percentile(border_distance, 75) + 4, 6, 15))
    candidate = np.linalg.norm(lab - background, axis=2) <= threshold

    height, width = candidate.shape
    connected = np.zeros((height, width), dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    for x in range(width):
        for y in (0, height - 1):
            if candidate[y, x] and not connected[y, x]:
                connected[y, x] = True
                queue.append((y, x))
    for y in range(height):
        for x in (0, width - 1):
            if candidate[y, x] and not connected[y, x]:
                connected[y, x] = True
                queue.append((y, x))
    while queue:
        y, x = queue.popleft()
        for neighbour_y, neighbour_x in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if (0 <= neighbour_y < height and 0 <= neighbour_x < width
                    and candidate[neighbour_y, neighbour_x]
                    and not connected[neighbour_y, neighbour_x]):
                connected[neighbour_y, neighbour_x] = True
                queue.append((neighbour_y, neighbour_x))
    pixels = rgb[~connected]
    return pixels if len(pixels) >= 100 else rgb.reshape(-1, 3)


def _cluster_lab(values: np.ndarray, clusters: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Small deterministic k-means implementation for colour pixels."""
    cluster_count = min(clusters, len(values))
    centres = [values[len(values) // 2]]
    for _ in range(1, cluster_count):
        distances = np.min(
            ((values[:, None, :] - np.asarray(centres)[None, :, :]) ** 2).sum(axis=2),
            axis=1,
        )
        centres.append(values[int(np.argmax(distances))])
    centres_array = np.asarray(centres)
    for _ in range(20):
        assignments = np.argmin(
            ((values[:, None, :] - centres_array[None, :, :]) ** 2).sum(axis=2),
            axis=1,
        )
        updated = np.asarray([
            values[assignments == index].mean(axis=0)
            if np.any(assignments == index) else centres_array[index]
            for index in range(cluster_count)
        ])
        if np.allclose(updated, centres_array, atol=0.05):
            centres_array = updated
            break
        centres_array = updated
    assignments = np.argmin(
        ((values[:, None, :] - centres_array[None, :, :]) ** 2).sum(axis=2),
        axis=1,
    )
    return centres_array, np.bincount(assignments, minlength=cluster_count)


def _calibrated_name(centre: np.ndarray, calibration: list[dict] | None) -> str:
    automatic = _palette_name(centre)
    if not calibration:
        return automatic
    valid = [entry for entry in calibration if entry.get("label") in PALETTE]
    if not valid:
        return automatic
    points = np.asarray([[entry["lab_l"], entry["lab_a"], entry["lab_b"]] for entry in valid])
    distances = np.linalg.norm(points - centre, axis=1)
    nearest = int(np.argmin(distances))
    return str(valid[nearest]["label"]) if distances[nearest] <= CALIBRATION_MAX_DISTANCE else automatic


def color_signature(image: Image.Image) -> tuple[float, float, float]:
    """Return the largest foreground cluster for user-specific calibration."""
    centres, counts = _cluster_lab(_rgb_to_lab(_garment_pixels(image)))
    centre = centres[int(np.argmax(counts))]
    return tuple(round(float(value), 4) for value in centre)


def infer_colors(
    image: Image.Image,
    limit: int = 3,
    calibration: list[dict] | None = None,
) -> list[tuple[str, float]]:
    """Return dominant named garment colours and their foreground shares."""
    if limit < 1:
        raise ValueError("Colour limit must be at least one.")
    pixels = _garment_pixels(image)
    lab_pixels = _rgb_to_lab(pixels)
    centres, counts = _cluster_lab(lab_pixels)
    shares: dict[str, float] = {}
    for centre, count in zip(centres, counts):
        share = float(count / counts.sum())
        if share < 0.03:
            continue
        name = _palette_name(centre)
        shares[name] = shares.get(name, 0.0) + share
    ranked = sorted(shares.items(), key=lambda entry: (-entry[1], entry[0]))
    dominant_centre = centres[int(np.argmax(counts))]
    calibrated = _calibrated_name(dominant_centre, calibration)
    if calibrated != _palette_name(dominant_centre):
        calibrated_share = shares.get(calibrated, float(counts.max() / counts.sum()))
        ranked = [(calibrated, calibrated_share)] + [entry for entry in ranked if entry[0] != calibrated]
    return [(name, round(share, 3)) for name, share in ranked[:limit]]


def infer_color(image: Image.Image, calibration: list[dict] | None = None) -> str:
    """Return the largest perceptual colour cluster for recommendations."""
    return infer_colors(image, limit=1, calibration=calibration)[0][0]
