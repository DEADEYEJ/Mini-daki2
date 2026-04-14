import cv2 as cv
import numpy as np
import os


def main():
    image_path = r"King Domino dataset\10.jpg" # der er brugt jpg nr. 1, 13, 45, 65, 22 og 42 til test af hsv
    if not os.path.isfile(image_path):
        print("Image not found")
        return
    image = cv.imread(image_path)
    tiles = get_tiles(image)
    print(len(tiles))
    for y, row in enumerate(tiles):
        for x, tile in enumerate(row):
            print(f"Tile ({x}, {y}):")
            print(get_terrain(tile))
            print("=====")


def get_tiles(image):
    tiles = []
    for y in range(5):
        tiles.append([])
        for x in range(5):
            tiles[-1].append(image[y*100:(y+1)*100, x*100:(x+1)*100])
    return tiles



def get_terrain(tile):
    hsv_tile = cv.cvtColor(tile, cv.COLOR_BGR2HSV)
    hue, saturation, value = np.median(hsv_tile, axis=(0,1)) # Consider using median instead of mean
    print(f"H: {hue}, S: {saturation}, V: {value}")
    if 35 < hue < 44 and 199 < saturation < 247 and 74 < value < 158:
        return "Grassland"
    
    if 19 < hue < 334 and 64 < saturation < 119 and 31 < value < 64:
        return "Mine"
    
    if 28 < hue < 58 and 83 < saturation < 206 and 34 < value < 63:
        return "Forest"
    
    if 105 < hue < 109 and 240 < saturation < 254 and 117 < value < 184:
        return "Lake"
    
    if 22 < hue < 46 and 232 < saturation < 254 and 135 < value < 201:
        return "Field"
    
    if 18 < hue < 23 and 75 < saturation < 166 and 82 < value < 118:
        return "Swamp"
    
    if 21 < hue < 87 and 41 < saturation < 194 and 59 < value < 120:
        return "Home"
    
    return "Unknown"

if __name__ == "__main__":
    main()
