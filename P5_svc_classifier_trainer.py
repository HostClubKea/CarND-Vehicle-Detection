from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.feature_extractor import FeatureExtractor
from sklearn.pipeline import Pipeline
from src.utils import *
import numpy as np

import time

from multiprocessing import Pool, freeze_support
from itertools import repeat

def extract_features(image):
    (w, h, d) = image.shape
    features = FeatureExtractor(image, trim_image=False).features([0,0,w,h])
    return features

if __name__ == '__main__':
    t1 = time.process_time()
    # Read training data
    print('Load training data...')
    cars = car_images()
    notcars = not_car_images()
    print('Training data info:')

    print('Vehicles images:', len(cars))
    print('Non-vehicles images:', len(notcars))

    # Extract features
    print('Extracting features...')
    car_features = []
    notcar_features = []
    (w,h,d) = cars[0].shape
    with Pool(3) as p:
        car_features = p.starmap(extract_features, zip(cars))

    elapsed_time1 = time.process_time() - t1
    print("\r\nElapsed time - " + str(elapsed_time1))

    with Pool(3) as p:
        notcar_features = p.starmap(extract_features, zip(notcars))

    elapsed_time1 = time.process_time() - t1
    print("\r\nElapsed time - " + str(elapsed_time1))

    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    print('    ...Done')

    print('Training classifier...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = Pipeline([('scaling', StandardScaler()),
                    ('classification', LinearSVC(loss='hinge'))])

    clf.fit(X_train, y_train)
    accuracy = round(clf.score(X_test, y_test), 4)
    print('    ...Done')
    print('Accuracy =', accuracy)

    elapsed_time1 = time.process_time() - t1
    print("\r\nTotal - " + str(elapsed_time1))

    save_pickle('svc_classifier.pkl', clf)
