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


def draw_ball(frame, position, trail=(), color=(255, 255, 255)):
    """Marker above the ball plus a fading trail of its recent positions."""

    points = [(int(round(x)), int(round(y))) for x, y in trail]
    for i in range(1, len(points)):
        strength = i / len(points)
        cv2.line(frame, points[i - 1], points[i], color, max(1, int(3 * strength)), cv2.LINE_AA)

    x, y = int(round(position[0])), int(round(position[1]))
    triangle = np.array([[x, y - 12], [x - 9, y - 28], [x + 9, y - 28]], dtype=np.int32)
    cv2.fillPoly(frame, [triangle], color, cv2.LINE_AA)
    cv2.polylines(frame, [triangle], True, (0, 0, 0), 1, cv2.LINE_AA)


def draw_possession_bar(frame, shares, colors, origin, size=(360, 26)):
    """Horizontal bar split by possession share. shares / colors: lists in the same team order."""

    x, y = origin
    width, height = size
    total = sum(shares)

    cv2.rectangle(frame, (x - 2, y - 2), (x + width + 2, y + height + 2), (20, 20, 20), -1)

    if total <= 0:
        cv2.putText(frame, "possession: -", (x + 8, y + height - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return

    left = x
    for share, color in zip(shares, colors):
        segment = int(round(width * share / total))
        if segment <= 0:
            continue
        cv2.rectangle(frame, (left, y), (left + segment, y + height), color, -1)
        if segment > 46:
            cv2.putText(frame, f"{share / total:.0%}", (left + 8, y + height - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, _text_color(color), 1, cv2.LINE_AA)
        left += segment


def display_color(lab_color):
    """A bright, saturated BGR color with the hue of a jersey color given in Lab."""

    lab = np.uint8([[np.clip(lab_color, 0, 255)]])
    hsv = cv2.cvtColor(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2HSV)[0, 0]
    hsv = np.uint8([[[hsv[0], max(int(hsv[1]), 170), 255]]])

    return tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])
