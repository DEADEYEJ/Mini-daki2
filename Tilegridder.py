import pandas as pd
import cv2 as cv
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

image_path = r"king Domino dataset\1.jpg"

df = pd.read_csv(r"tiles_HSV.csv", sep=",")
df.dropna(subset=['image', 'tile_x', 'tile_y', 'hue', 'saturation', 'value', 'label'], inplace=True)

X = df[['hue', 'saturation', 'value']].values
y = df['label'].values

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

k = 5
knn_model = KNeighborsClassifier(n_neighbors=k)
knn_model.fit(x_train, y_train)

y_pred = knn_model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
#print("Accuracy:", accuracy * 100, "%")

def print_grid(grid):
    print("\nPredicted Terrain Grid (5x5):\n")
    for row in grid:
        print(" | ".join(f"{cell:^10}" for cell in row))

def tilesCut(image):
    tiles = []
    for y in range(5):
        tiles.append([])
        for x in range(5):
            tiles[-1].append(image[y*100:(y+1)*100, x*100:(x+1)*100])
    return tiles

def pred_terrain(tile):
    tile_hsv = cv.cvtColor(tile, cv.COLOR_BGR2HSV)
    hue, saturation, value = np.median(tile_hsv, axis=(0,1))
    feature = np.array([[hue, saturation, value]])
    #print(f"Hue: {hue}, Saturation: {saturation}, Value: {value}")
    pred_knn = knn_model.predict(feature)
    return pred_knn[0]

def terrain_grid(image_path):
    IMG = cv.imread(image_path)
    tiles = tilesCut(IMG)

    grid = [
        [str(pred_terrain(tile)) for tile in row]
        for row in tiles
    ]
    return grid

def main():
    grid = terrain_grid(image_path)
    print_grid(grid)

if __name__ == "__main__":
    main()




