import cv2
import numpy as np

from src.pitch.drawing import PitchDiagram


def draw_player(frame, box, color, label=None):
    """Ellipse at the player's feet plus an optional label tag above the head."""

    x1, y1, x2, y2 = (int(round(v)) for v in box)
    center_x = (x1 + x2) // 2
    width = max(x2 - x1, 12)

    cv2.ellipse(
        frame, (center_x, y2), (width // 2 + 4, max(width // 5, 4)),
        0, -45, 235, color, 2, cv2.LINE_AA,
    )

    if label:
        font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
        (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
        tx, ty = center_x - tw // 2, y1 - 6

        cv2.rectangle(frame, (tx - 3, ty - th - 4), (tx + tw + 3, ty + 3), color, -1)
        cv2.putText(frame, label, (tx, ty), font, scale, _text_color(color), thickness, cv2.LINE_AA)


def draw_minimap(frame, diagram: PitchDiagram, points, colors, origin, alpha=0.85):
    """Paste a top-down pitch with player dots onto the frame at origin (x, y)."""

    radar = diagram.image.copy()

    for (px, py), color in zip(diagram.to_px(points), colors):
        if np.isfinite([px, py]).all():
            center = (int(round(px)), int(round(py)))
            cv2.circle(radar, center, 6, color, -1, cv2.LINE_AA)
            cv2.circle(radar, center, 6, (0, 0, 0), 1, cv2.LINE_AA)

    x, y = origin
    h, w = radar.shape[:2]
    region = frame[y:y + h, x:x + w]

    if region.shape[:2] != (h, w):
        return

    cv2.addWeighted(radar, alpha, region, 1 - alpha, 0, dst=region)
    cv2.rectangle(frame, (x, y), (x + w - 1, y + h - 1), (255, 255, 255), 2)


def draw_heatmap(diagram: PitchDiagram, points, sigma_m=3.0, colormap=cv2.COLORMAP_JET):
    """Density heatmap of pitch positions (meters) on top of the pitch diagram."""

    image = diagram.image.copy()
    density = np.zeros(image.shape[:2], dtype=np.float32)

    for px, py in diagram.to_px(points):
        ix, iy = int(round(px)), int(round(py))
        if 0 <= iy < density.shape[0] and 0 <= ix < density.shape[1]:
            density[iy, ix] += 1

    if density.max() == 0:
        return image

    density = cv2.GaussianBlur(density, (0, 0), sigma_m * diagram.scale)
    density /= density.max()

    colored = cv2.applyColorMap((density * 255).astype(np.uint8), colormap)
    weight = np.clip(density * 1.5, 0, 0.75)[..., None]

    return (image * (1 - weight) + colored * weight).astype(np.uint8)


def _text_color(background):
    b, g, r = background
    return (0, 0, 0) if 0.299 * r + 0.587 * g + 0.114 * b > 140 else (255, 255, 255)
