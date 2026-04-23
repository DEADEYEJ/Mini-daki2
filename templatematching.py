import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# =========================
# INDSTILLINGER
# =========================
IMAGE_PATH = "King Domino dataset/73.jpg"
TEMPLATES_DIR = "templates"
CSV_PATH = "tiles_hsv.csv"   # valgfri, sæt til None hvis du ikke vil evaluere
MATCH_THRESHOLD = 0.91
MIN_DISTANCE = 10            # minimum afstand mellem to kroner i samme tile
GRID_SIZE = 5

VALID_TEMPLATE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


# =========================
# TILE OPDELING
# =========================
def get_tiles(image):
    tiles = []

    height, width, _ = image.shape
    tile_h = height // GRID_SIZE
    tile_w = width // GRID_SIZE

    for y in range(GRID_SIZE):
        row = []
        for x in range(GRID_SIZE):
            tile = image[y*tile_h:(y+1)*tile_h, x*tile_w:(x+1)*tile_w]
            row.append(tile)
        tiles.append(row)

    return tiles


# =========================
# TEMPLATE LOADING
# =========================
def load_template(path):
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"Kunne ikke læse template: {path}")

    if raw.ndim == 2:
        bgr = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        mask = None
    elif raw.shape[2] == 4:
        bgr = raw[:, :, :3]
        alpha = raw[:, :, 3]
        mask = np.where(alpha > 0, 255, 0).astype(np.uint8)
        if cv2.countNonZero(mask) == 0:
            mask = None
    else:
        bgr = raw[:, :, :3]
        mask = None

    return bgr, mask


def load_templates(folder):
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Template-mappen findes ikke: {folder}")

    template_paths = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_TEMPLATE_SUFFIXES
    )

    if not template_paths:
        raise FileNotFoundError(f"Ingen templates fundet i: {folder}")

    templates = []
    for path in template_paths:
        tpl_bgr, tpl_mask = load_template(path)
        templates.append({
            "name": path.name,
            "image": tpl_bgr,
            "mask": tpl_mask
        })

    return templates


# =========================
# HJÆLPEFUNKTIONER
# =========================
def preprocess_gray(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return gray


def point_distance(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def non_max_peaks(peaks, min_distance):
    """
    peaks: liste af dicts med x, y, score, w, h
    Beholder de stærkeste peaks og fjerner peaks der ligger for tæt.
    """
    peaks = sorted(peaks, key=lambda p: p["score"], reverse=True)
    kept = []

    for peak in peaks:
        center = (peak["cx"], peak["cy"])
        too_close = False

        for kept_peak in kept:
            kept_center = (kept_peak["cx"], kept_peak["cy"])
            if point_distance(center, kept_center) < min_distance:
                too_close = True
                break

        if not too_close:
            kept.append(peak)

    return kept


# =========================
# MATCHING I EN TILE
# =========================
def find_crowns_in_tile(tile_bgr, templates, threshold=0.82, min_distance=10):
    tile_gray = preprocess_gray(tile_bgr)
    all_peaks = []

    for tpl in templates:
        tpl_bgr = tpl["image"]
        tpl_mask = tpl["mask"]
        tpl_gray = preprocess_gray(tpl_bgr)

        th, tw = tpl_gray.shape[:2]
        ih, iw = tile_gray.shape[:2]

        # spring over hvis template er større end tile
        if th > ih or tw > iw:
            continue

        if tpl_mask is not None:
            result = cv2.matchTemplate(tile_gray, tpl_gray, cv2.TM_CCORR_NORMED, mask=tpl_mask)
        else:
            result = cv2.matchTemplate(tile_gray, tpl_gray, cv2.TM_CCORR_NORMED)

        ys, xs = np.where(result >= threshold)

        for x, y in zip(xs, ys):
            score = float(result[y, x])
            peak = {
                "x": int(x),
                "y": int(y),
                "w": int(tw),
                "h": int(th),
                "cx": int(x + tw / 2),
                "cy": int(y + th / 2),
                "score": score,
                "template": tpl["name"]
            }
            all_peaks.append(peak)

    final_peaks = non_max_peaks(all_peaks, min_distance=min_distance)
    return final_peaks


# =========================
# KØR PÅ HELE BILLEDET
# =========================
def count_crowns_per_tile(image_bgr, templates, threshold=0.82, min_distance=10):
    tiles = get_tiles(image_bgr)

    counts = []
    detections_per_tile = []

    for tile_y, row in enumerate(tiles):
        count_row = []
        det_row = []

        for tile_x, tile in enumerate(row):
            detections = find_crowns_in_tile(
                tile,
                templates,
                threshold=threshold,
                min_distance=min_distance
            )
            count_row.append(len(detections))
            det_row.append(detections)

        counts.append(count_row)
        detections_per_tile.append(det_row)

    return counts, detections_per_tile


# =========================
# TEGN RESULTAT
# =========================
def draw_tile_results(image_bgr, counts, detections_per_tile):
    out = image_bgr.copy()
    height, width, _ = out.shape
    tile_h = height // GRID_SIZE
    tile_w = width // GRID_SIZE

    for tile_y in range(GRID_SIZE):
        for tile_x in range(GRID_SIZE):
            x0 = tile_x * tile_w
            y0 = tile_y * tile_h

            # tegn tile-ramme
            cv2.rectangle(out, (x0, y0), (x0 + tile_w, y0 + tile_h), (255, 0, 0), 1)

            # skriv antal kroner i tile
            label = str(counts[tile_y][tile_x])
            cv2.putText(
                out,
                label,
                (x0 + 5, y0 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

            # tegn fund inde i tile
            for det in detections_per_tile[tile_y][tile_x]:
                dx1 = x0 + det["x"]
                dy1 = y0 + det["y"]
                dx2 = dx1 + det["w"]
                dy2 = dy1 + det["h"]

                cv2.rectangle(out, (dx1, dy1), (dx2, dy2), (0, 0, 255), 1)

    return out


# =========================
# CSV EVALUERING
# =========================
def evaluate_against_csv(counts, csv_path, image_name):
    df = pd.read_csv(csv_path)

    required = {"image", "tile_x", "tile_y", "crown"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV mangler kolonner: {sorted(missing)}")

    df = df[df["image"] == image_name].copy()
    if df.empty:
        raise ValueError(f"Ingen rækker i CSV for image='{image_name}'")

    predicted = []
    actual = []

    for _, row in df.iterrows():
        tx = int(row["tile_x"])
        ty = int(row["tile_y"])

        if 0 <= ty < GRID_SIZE and 0 <= tx < GRID_SIZE:
            predicted_count = counts[ty][tx]
            actual_count = int(row["crown"])

            predicted.append(predicted_count)
            actual.append(actual_count)

    if not actual:
        raise ValueError("Ingen gyldige tiles at evaluere")

    exact_matches = sum(int(p == a) for p, a in zip(predicted, actual))
    exact_tile_accuracy = exact_matches / len(actual)

    mae = np.mean([abs(p - a) for p, a in zip(predicted, actual)])

    print("\nCSV evaluering pr. tile:")
    print(f"Antal tiles evalueret: {len(actual)}")
    print(f"Exact tile accuracy: {exact_tile_accuracy:.4f}")
    print(f"MAE (gennemsnitlig fejl i crown count): {mae:.4f}")

    print("\nDetaljer:")
    for a, p in zip(actual, predicted):
        print(f"actual={a}, predicted={p}")


# =========================
# MAIN
# =========================
def main():
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        raise FileNotFoundError(f"Kunne ikke læse billede: {IMAGE_PATH}")

    templates = load_templates(TEMPLATES_DIR)
    print(f"Loaded {len(templates)} templates")

    counts, detections_per_tile = count_crowns_per_tile(
        image,
        templates,
        threshold=MATCH_THRESHOLD,
        min_distance=MIN_DISTANCE
    )

    print("\nKroner pr. tile:")
    for row in counts:
        print(row)

    out = draw_tile_results(image, counts, detections_per_tile)
    cv2.imwrite("tile_crown_counts.png", out)
    print("\nGemte output-billede som: tile_crown_counts.png")

    if CSV_PATH is not None:
        evaluate_against_csv(counts, CSV_PATH, Path(IMAGE_PATH).name)


if __name__ == "__main__":
    main()