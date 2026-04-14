import cv2 as cv
import numpy as np
import os
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# Her indlæser vi træningsdata (samme datasæt som i knn_model_train.py)
# Datasættet virkede kun ved semikolonspereret csv-fil
df = pd.read_csv(r'C:\Users\danie\Desktop\python_work\P0---gruppe-4\data_KD.csv', sep=';')

# Vi fjerner de tomme felter
df.dropna(subset=['h', 's', 'v', 'target'], inplace=True)

# Vi tildeler X og y vores data værdier
X = df[['h', 's', 'v']].values
y = df['target'].values

# Her fitter vi KNN modellen med vores datasæt og sætter k=11, da dette var det bedste valg i knn_model_train.py
k = 11
knn_classifier = KNeighborsClassifier(n_neighbors=k)
knn_classifier.fit(X, y)

# Selve hoveddelen af programmet, hvor den printer de forskellgie info og indlæser et billede
def main():
    print("+-------------------------------+")
    print("| King Domino points calculator |")
    print("+-------------------------------+")
    image_path = r"C:\Users\danie\Downloads\King Domino dataset\57.jpg"
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

# Her opdeler den billedet i 5x5 felter (100x100 pixels hver)
def get_tiles(image):
    tiles = []
    for y in range(5):
        tiles.append([])
        for x in range(5):
            tiles[-1].append(image[y*100:(y+1)*100, x*100:(x+1)*100])
    return tiles

# Her bruger vi vores KNN model til at forudsige terræntypen baseret på medianværdierne af HSV
def get_terrain_knn(tile):
    hsv_tile = cv.cvtColor(tile, cv.COLOR_BGR2HSV)
    hue, saturation, value = np.median(hsv_tile, axis=(0,1))
    features = np.array([[hue, saturation, value]])
    print(f"Hue: {hue}, Saturation: {saturation}, Value: {value}")
    prediction = knn_classifier.predict(features)
    return prediction[0]

if __name__ == "__main__":
    main()
