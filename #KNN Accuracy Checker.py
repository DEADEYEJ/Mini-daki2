#KNN Accuracy checker
import numpy as np
import pandas as pd
import cv2 as cv
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv(r"Kingdomino-HSV.csv", sep=";")
df.dropna(subset=['h', 's', 'v', 'target'], inplace=True)

X = df[['h','s','v']].values
y = df['target'].values

k = 11
knn_classifier = KNeighborsClassifier(n_neighbors=k)
knn_classifier.fit(X, y)

def main():
    image_path = r"King Domino dataset\50.jpg"
    if not os.path.isfile(image_path):
        print("Image not found")
        return
    image = cv.imread(image_path)
    tiles = get_tiles(image)
    print(len(tiles))
    for y, row in enumerate(tiles):
        for x, tile in enumerate(row):
            print(f"Tile ({x}, {y}):")
            print(get_terrain_knn(tile))
            print("=====")

def get_tiles(image):
    tiles = []
    for y in range(5):
        tiles.append([])
        for x in range(5):
            tiles[-1].append(image[y*100:(y+1)*100, x*100:(x+1)*100])
    return tiles

def get_terrain_knn(tile):
    hsv_tile = cv.cvtColor(tile, cv.COLOR_BGR2HSV)
    hue, saturation, value = np.median(hsv_tile, axis=(0,1))
    features = np.array([[hue, saturation, value]])
    print(f"Hue: {hue}, Saturation: {saturation}, Value: {value}")
    prediction = knn_classifier.predict(features)
    return prediction[0]

if __name__ == "__main__":
    main()
