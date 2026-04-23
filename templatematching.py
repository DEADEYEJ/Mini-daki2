#!/usr/bin/env python3
"""
Template matching for crowns / markers with support for multiple rotations.

Usage:
    python template_match_crowns.py \
        --image 2.jpg \
        --templates crown_0.png crown_90.png crown_180.png crown_270.png \
        --threshold 0.86 \
        --output detections.png \
        --json detections.json

Notes:
- PNG templates with transparency are supported. The alpha channel is used as a mask.
- The script runs template matching for each template and merges overlapping detections.
- Start with a threshold around 0.82-0.90 and adjust based on your data.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def load_template(path: str):
    """Load template and optional alpha mask."""
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"Could not read template: {path}")

    if raw.ndim == 2:
        rgb = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        mask = None
    elif raw.shape[2] == 4:
        rgb = raw[:, :, :3]
        alpha = raw[:, :, 3]
        # Binary-ish mask; ignore fully transparent pixels
        mask = np.where(alpha > 0, 255, 0).astype(np.uint8)
        if cv2.countNonZero(mask) == 0:
            mask = None
    else:
        rgb = raw[:, :, :3]
        mask = None

    return rgb, mask


def match_single_template(image_bgr, template_bgr, mask, threshold, template_name):
    """Run masked template matching and return raw detections."""
    img_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    tpl_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)

    # Light normalization helps on screenshots/photos with different lighting
    img_gray = cv2.equalizeHist(img_gray)
    tpl_gray = cv2.equalizeHist(tpl_gray)

    # TM_CCORR_NORMED works well with masks in OpenCV
    if mask is not None:
        result = cv2.matchTemplate(img_gray, tpl_gray, cv2.TM_CCORR_NORMED, mask=mask)
    else:
        result = cv2.matchTemplate(img_gray, tpl_gray, cv2.TM_CCORR_NORMED)

    ys, xs = np.where(result >= threshold)
    h, w = tpl_gray.shape[:2]

    detections = []
    for (x, y) in zip(xs, ys):
        score = float(result[y, x])
        detections.append({
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
            "score": score,
            "template": Path(template_name).name,
            "center_x": int(x + w / 2),
            "center_y": int(y + h / 2),
        })

    return detections, result


def iou(a, b):
    ax1, ay1 = a["x"], a["y"]
    ax2, ay2 = ax1 + a["w"], ay1 + a["h"]
    bx1, by1 = b["x"], b["y"]
    bx2, by2 = bx1 + b["w"], by1 + b["h"]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def non_max_suppression(detections, overlap_threshold=0.25):
    """Keep highest-score boxes when detections overlap."""
    detections = sorted(detections, key=lambda d: d["score"], reverse=True)
    kept = []

    while detections:
        best = detections.pop(0)
        kept.append(best)
        remaining = []
        for det in detections:
            if iou(best, det) < overlap_threshold:
                remaining.append(det)
        detections = remaining

    return kept


def draw_detections(image_bgr, detections):
    out = image_bgr.copy()
    for i, det in enumerate(detections, start=1):
        x, y, w, h = det["x"], det["y"], det["w"], det["h"]
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 2)
        label = f'{i}: {det["score"]:.2f}'
        cv2.putText(
            out, label, (x, max(18, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA
        )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument(
        "--templates", nargs="+", required=True,
        help="One or more template files, e.g. 4 rotations"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.86,
        help="Match threshold, usually 0.82-0.90"
    )
    parser.add_argument(
        "--nms", type=float, default=0.25,
        help="IoU threshold for non-max suppression"
    )
    parser.add_argument(
        "--output", default="detections.png",
        help="Output image with marked detections"
    )
    parser.add_argument(
        "--json", default="detections.json",
        help="Output json file"
    )
    args = parser.parse_args()

    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    all_detections = []
    per_template_max = {}

    for template_path in args.templates:
        template_bgr, mask = load_template(template_path)
        detections, result = match_single_template(
            image, template_bgr, mask, args.threshold, template_path
        )
        all_detections.extend(detections)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        per_template_max[Path(template_path).name] = {
            "max_score": float(max_val),
            "best_location": [int(max_loc[0]), int(max_loc[1])]
        }

    final_detections = non_max_suppression(all_detections, overlap_threshold=args.nms)

    marked = draw_detections(image, final_detections)
    cv2.imwrite(args.output, marked)

    payload = {
        "image": str(args.image),
        "templates": [str(t) for t in args.templates],
        "threshold": args.threshold,
        "nms_iou_threshold": args.nms,
        "count": len(final_detections),
        "detections": final_detections,
        "best_match_per_template": per_template_max,
    }

    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Saved marked image to: {args.output}")
    print(f"Saved detections json to: {args.json}")
    print(f"Found {len(final_detections)} detections")
    for i, det in enumerate(final_detections, start=1):
        print(
            f'#{i}: template={det["template"]}, score={det["score"]:.3f}, '
            f'x={det["x"]}, y={det["y"]}, center=({det["center_x"]}, {det["center_y"]})'
        )


if __name__ == "__main__":
    main()
