import numpy as np
import pandas as pd
import cv2 as cv
import os
from skimage.feature import hog
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

IMAGE_ROOT = r"King Domino dataset"
TILE_SIZE = 100
HOG_SIZE = (64, 64)


def get_tiles(image):
    tiles = []
    for y in range(5):
        tiles.append([])
        for x in range(5):
            tiles[-1].append(image[y * TILE_SIZE:(y + 1) * TILE_SIZE, x * TILE_SIZE:(x + 1) * TILE_SIZE])
    return tiles


def extract_hog_features(tile):
    gray = cv.cvtColor(tile, cv.COLOR_BGR2GRAY)
    gray = cv.resize(gray, HOG_SIZE)

    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True
    )
    return features


def build_training_set(df):
    X = []
    y = []

    for _, row in df.iterrows():
        image_path = os.path.join(IMAGE_ROOT, str(row["image"]))
        if not os.path.isfile(image_path):
            continue

        image = cv.imread(image_path)
        if image is None:
            continue

        tile_x = int(row["tile_x"])
        tile_y = int(row["tile_y"])

        tile = image[
            tile_y * TILE_SIZE:(tile_y + 1) * TILE_SIZE,
            tile_x * TILE_SIZE:(tile_x + 1) * TILE_SIZE
        ]

        if tile.size == 0:
            continue

        features = extract_hog_features(tile)
        X.append(features)
        y.append(row["target"])

    return np.array(X), np.array(y)


def train_classifier():
    df = pd.read_csv(r"Kingdomino-tiles.csv", sep=";")
    df.dropna(subset=["image", "tile_x", "tile_y", "target"], inplace=True)

    X, y = build_training_set(df)

    if len(X) == 0:
        print("Ingen træningsdata fundet.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    knn_classifier = KNeighborsClassifier(n_neighbors=11)
    knn_classifier.fit(X_train, y_train)

    y_pred = knn_classifier.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return knn_classifier


def get_terrain_hog(tile, classifier):
    features = extract_hog_features(tile)
    prediction = classifier.predict([features])
    return prediction[0]


def main():
    classifier = train_classifier()
    if classifier is None:
        return

    image_path = r"King Domino dataset\50.jpg"
    if not os.path.isfile(image_path):
        print("Image not found")
        return

    image = cv.imread(image_path)
    tiles = get_tiles(image)

    for y, row in enumerate(tiles):
        for x, tile in enumerate(row):
            print(f"Tile ({x}, {y}): {get_terrain_hog(tile, classifier)}")
            print("=====")


if __name__ == "__main__":
    main()