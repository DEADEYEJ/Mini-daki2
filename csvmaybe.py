import cv2 as cv
import numpy as np
import os
import csv
from collections import Counter


def main():
    folder_path = r"King Domino dataset"
    output_csv = "tiles_hsv.csv"

    if not os.path.isdir(folder_path):
        print("Folder not found")
        return

    image_files = [f for f in os.listdir(folder_path) if f.endswith(".jpg")]

    label_counts = Counter()

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)

        # CSV header
        writer.writerow(["image", "tile_x", "tile_y", "hue", "saturation", "value", "label"])

        for image_name in image_files:
            image_path = os.path.join(folder_path, image_name)
            image = cv.imread(image_path)

            if image is None:
                print(f"Could not load {image_name}")
                continue

            tiles = get_tiles(image)

            for y, row in enumerate(tiles):
                for x, tile in enumerate(row):
                    h, s, v = get_terrain(tile)
                    label = classify_tile(h, s, v)

                    writer.writerow([image_name, x, y, h, s, v, label])
                    label_counts[label] += 1

    print(f"CSV file saved as {output_csv}")
    print("\nLabel distribution:")
    for label, count in label_counts.items():
        print(f"{label}: {count}")


def get_tiles(image):
    tiles = []

    height, width, _ = image.shape
    tile_h = height // 5
    tile_w = width // 5

    for y in range(5):
        row = []
        for x in range(5):
            tile = image[y*tile_h:(y+1)*tile_h, x*tile_w:(x+1)*tile_w]
            row.append(tile)
        tiles.append(row)

    return tiles


def get_terrain(tile):
    hsv_tile = cv.cvtColor(tile, cv.COLOR_BGR2HSV)

    # Fjern pixels med meget lav saturation (typisk baggrund/støj)
    mask = hsv_tile[:, :, 1] > 20
    filtered = hsv_tile[mask]

    if len(filtered) == 0:
        filtered = hsv_tile.reshape(-1, 3)

    hue, saturation, value = np.median(filtered, axis=0)

    return int(hue), int(saturation), int(value)


def classify_tile(hue, saturation, value):
    if 22 <= hue <= 27 and 219 <= saturation <= 255 and 135 <= value <= 206:
        return "Field"

    if 104 <= hue <= 109 and 222 <= saturation <= 255 and 108 <= value <= 204:
        return "Lake"

    if 35 <= hue <= 51 and 156 <= saturation <= 248 and 74 <= value <= 164:
        return "Grassland"

    if 19 <= hue <= 30 and 34 <= saturation <= 147 and 24 <= value <= 82:
        return "Mine"

    if 28 <= hue <= 77 and 65 <= saturation <= 225 and 25 <= value <= 78:
        return "Forest"

    if 17 <= hue <= 29 and 23 <= saturation <= 180 and 73 <= value <= 144:
        return "Swamp"

    if 16 <= hue <= 40 and 28 <= saturation <= 194 and 35 <= value <= 150:
        return "Home"

    return "Unknown"


if __name__ == "__main__":
    main()