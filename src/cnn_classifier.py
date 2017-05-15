from src.utils import load_pickle
from keras.models import load_model
import numpy as np
import random


class CnnClassifier():

    def __init__(self, model_file='model.h5'):
        self.image = None
        self.feature_extractor = None
        self.model = load_model(model_file)

    def classify(self, image, windows):
        self.image = image
        if np.issubdtype(self.image.dtype, np.integer):
            self.image = (self.image / 255).astype(np.float32)

        X = []

        for window in windows:
            x = self.image[window[1]:window[3], window[0]:window[2], :]
            X.append(x)

        X = np.array(X)
        predictions = self.model.predict_classes(X, batch_size=128, verbose=0)
        return predictions


