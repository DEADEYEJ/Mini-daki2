import cv2 as cv
import numpy as np
import os
import csv


def main():
    folder_path = r"King Domino dataset"
    output_csv = "tiles_hsv.csv"

    if not os.path.isdir(folder_path):
        print("Folder not found")
        return

    # Find alle billeder
    image_files = [f for f in os.listdir(folder_path) if f.endswith(".jpg")]

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Header
        writer.writerow(["image", "tile_x", "tile_y", "hue", "saturation", "value"])

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
                    writer.writerow([image_name, x, y, h, s, v])

    print(f"CSV file saved as {output_csv}")


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

    # Median er mere robust mod støj end mean
    hue, saturation, value = np.median(hsv_tile, axis=(0, 1))

    return int(hue), int(saturation), int(value)


if __name__ == "__main__":
    main()