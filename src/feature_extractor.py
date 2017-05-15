import numpy as np
import cv2
from skimage.feature import hog

class FeatureExtractor():

    def __init__(self, image, orient=10, pix_per_cell=8, cell_per_block=2, trim_image = True, trim_rate = 0.5):
        self.image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        if np.issubdtype(self.image.dtype, np.integer):
            self.image = ( self.image / 255).astype(np.float32)

        (self.h, self.w, self.d) = self.image.shape

        self.trim_image = trim_image
        self.trim_rate = trim_rate

        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.hog_features = []
        self.hog()

    # Calculate hog for whole image
    def hog(self):

        for channel in range(self.d):
            if self.trim_image:
                self.hog_features.append(
                    hog(self.image[int(self.h*self.trim_rate):, :, channel], orientations=self.orient,
                        pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                        cells_per_block=(self.cell_per_block, self.cell_per_block), transform_sqrt=True,
                        visualise=False, feature_vector=False)
                )
            else:
                self.hog_features.append(
                    hog(self.image[:, :, channel], orientations=self.orient,
                        pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                        cells_per_block=(self.cell_per_block, self.cell_per_block), transform_sqrt=True,
                        visualise=False, feature_vector=False)
                )
        self.hog_features = np.asarray(self.hog_features)

    # Extract hog features for window
    def window_hog(self, window):
        # print(window[2] - window[0])
        if self.trim_image:
            hog_k = ((window[2] - window[0]) // self.pix_per_cell) - 1
            hog_x = max((window[0] // self.pix_per_cell) - 1, 0)
            hog_x = self.hog_features.shape[2] - hog_k if hog_x + hog_k > self.hog_features.shape[2] else hog_x
            hog_y = max(((window[1] - int(self.h*self.trim_rate)) // self.pix_per_cell) - 1, 0)
            hog_y = self.hog_features.shape[1] - hog_k if hog_y + hog_k > self.hog_features.shape[1] else hog_y

            return np.ravel(self.hog_features[:, hog_y:hog_y + hog_k, hog_x:hog_x + hog_k, :, :, :])
        else:
            hog_k = ((window[2] - window[0]) // self.pix_per_cell) - 1
            hog_x = max((window[0] // self.pix_per_cell) - 1, 0)
            hog_x = self.hog_features.shape[2] - hog_k if hog_x + hog_k > self.hog_features.shape[2] else hog_x
            hog_y = max((window[1] // self.pix_per_cell) - 1, 0)
            hog_y = self.hog_features.shape[1] - hog_k if hog_y + hog_k > self.hog_features.shape[1] else hog_y

            return np.ravel(self.hog_features[:, hog_y:hog_y+hog_k, hog_x:hog_x+hog_k, :, :, :])

        #
        # # Get width and heght of window
        # dx = window[2] - window[0]
        # dy = window[3] - window[1]
        # print(dx)
        # # Get width and height for hog features
        # hog_dx = int(dx/self.pix_per_cell)
        # hog_dy = int(dy/self.pix_per_cell)
        #
        # hog_x = int(window[0] / self.pix_per_cell)
        # hog_y = int(window[1] / self.pix_per_cell)
        #
        # return np.ravel(self.hog_features[:, hog_y:hog_y+hog_dy, hog_x:hog_x+hog_dx, :, :, :])

    # Function to compute binned color features
    def bin_spatial(self, img, size=(16, 16)): #16*16
        # Use cv2.resize().ravel() to create the feature vector
        features = cv2.resize(img, size).ravel()
        # Return the feature vector
        return features

    # Function to compute color histogram features
    def color_hist(self, img, nbins=16, bins_range=(0, 256)):#nbins=16
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
        channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
        channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        # Return the individual histograms, bin_centers and feature vector
        return hist_features

    def features(self, window):
        features = []
        window_image = self.image[window[1]:window[3], window[0]:window[2], :]

        # Extract binned color features

        spatial_features = self.bin_spatial(window_image)
        features.append(spatial_features)

        # Extract histogram features
        hist_features = self.color_hist(window_image)
        features.append(hist_features)
        # print(len(hist_features))

        hog_features = self.window_hog(window)
        features.append(hog_features)

        # print(len(hog_features))
        # Combine features into single vector
        return np.concatenate(features)
