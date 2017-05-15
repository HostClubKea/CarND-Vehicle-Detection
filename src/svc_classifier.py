from src.feature_extractor import FeatureExtractor
from src.utils import load_pickle
import numpy as np
import random


class SvcClassifier():

    def __init__(self, classifier='svc_classifier.pkl'):
        self.image = None
        self.feature_extractor = None
        self.classifier = load_pickle(classifier)

    def classify(self, image, windows):
        self.image = image
        self.feature_extractor = FeatureExtractor(image)
        features = []
        for window in windows:
            features.append(self.feature_extractor.features(window))

        return self.classifier.predict(features)
