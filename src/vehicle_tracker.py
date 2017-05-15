from collections import deque
from scipy.ndimage.measurements import label
import numpy as np
import cv2

from multiprocessing import Pool, freeze_support
from itertools import repeat
# 1. Windows
# 2. Display windows

class VehicleTracker():
    # Init vehicle tracker
    def __init__(self, windows_slider, classifiers, continuous = True, h=720, w=1280):
        # Image dimensions
        self.h = h
        self.w = w
        self.windows_slider = windows_slider
        # List of classifiers - svn, cnn
        self.classifiers = classifiers
        # Continuous mode
        self.continuous = continuous
        # Detection history
        self.detections_history = deque(maxlen=20)
        # Detections
        # self.detections = None
        self.heatmap = None
        pass

    def process(self, image, draw_detections = True):
        if not self.continuous:
            self.detections_history.clear()

        self.detect_vehicles(image)
        # self.windows_slider.draw(image)
        # if draw_detections:
        #     for c in self.detections():
        #         cv2.rectangle(image, (c[0], c[1]), (c[2], c[3]), (0, 0, 255), 2)

        image = self.heatmap
        return image

    def detections(self):
        detections, _ = self.process_detections(
            np.concatenate(np.array(self.detections_history)),
            threshold=min(len(self.detections_history), 15)
        )
        return detections

    def detect_vehicles(self, image):
        windows_ = self.windows_slider.windows(image)
        detections = np.empty([0, 4], dtype=np.int)
        for (scaled_image, scale, windows) in windows_:
            for classifier in self.classifiers:
                scale_detections = []

                predictions = classifier.classify(scaled_image, windows)

                for prediction, window in zip(np.array(predictions), windows):
                    if prediction == 1:
                        scale_detections.append(window)

                scale_detections = (np.array(scale_detections) / scale).astype(np.int)

                # for c in scale_detections:
                #     cv2.rectangle(image, (c[0], c[1]), (c[2], c[3]), (0, 0, 255), 2)

                if(len(scale_detections) > 0):
                    detections = np.append(detections, scale_detections, axis=0)
        detections, self.heatmap = self.process_detections(detections)
        self.detections_history.append(detections)


    def process_detections(self, detections, threshold=1):
        #Init empty heatmap array
        heatmap = np.zeros((self.h, self.w)).astype(np.float)
        # Add heat to each box in box list
        heatmap = self.add_heat(heatmap, detections)
        # Apply threshold to help remove false positives
        heatmap[heatmap < threshold] = 0
        heatmap = np.clip(heatmap, 0, 255)
        labels = label(heatmap)


        cars = np.empty([0, 4], dtype=np.int64)
        # Iterate through all detected cars
        for car in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car).nonzero()
            cars = np.append(
                cars,
                [[np.min(nonzero[1]), np.min(nonzero[0]), np.max(nonzero[1]), np.max(nonzero[0])]],
                axis=0
            )
        # Return the image

        return (cars, heatmap)


    def add_heat(self, heatmap, windows):
        # Iterate through list of windows
        for window in windows:
            # Add += 1 for all pixels inside each window
            # Assuming each "box" takes the form [x1, y1, x2, y2]
            heatmap[window[1]:window[3], window[0]:window[2]] += 1
            # Return updated heatmap
        return heatmap

    def draw_detections(self, image, color=(0, 0, 255), width=2):
        for c in self.detections:
            cv2.rectangle(image, (c[0], c[1]), (c[2], c[3]), color, width)
        return image
