import numpy as np
import cv2
from scipy.misc import imresize
from skimage.transform import resize

class WindowsSlider():
    def __init__(self, scales=[.3, .5, .65, .8], y_positions=[.6, .57, .56, .55], overlaps=[0.66, 0.66, 0.66, 0.66], window=(64, 64), h=720, w=1280, d=3): #[0.3, 0.5, 0.65, 0.8] [0.55, 0.55, 0.55, 0.55] [0.5, 0.75, 0.25, 0.25]
        self.h = h
        self.w = w
        self.d = d
        self.scales = scales
        self.y_positions = y_positions
        self.overlaps = overlaps
        self.window = window
        self.generated_windows = []
        # Windows could be pregenerated as parameters don't change
        # Might be usefull in case of using multithreading detection
        self.generate_windows()

    # Generate windows positions
    def generate_windows(self):
        for scale, overlap, position in zip(self.scales, self.overlaps, self.y_positions):
            # Scaled height
            h = int(self.h * scale)
            # Scaled width
            w = int(self.w * scale)
            # Generate windows
            y_start = int(position * h)
            scale_windows = self.slide_windows(h, w, x_start_stop=[0, w], y_start_stop=[y_start, min(h, y_start + int(64))], xy_overlap=(overlap, overlap)) #1.5

            self.generated_windows.append(scale_windows)

    def slide_windows(self, h, w, x_start_stop=[None, None], y_start_stop=[None, None],
                      xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
        # If x and/or y start/stop positions not defined, set to image size
        if x_start_stop[0] is None:
            x_start_stop[0] = 0
        if x_start_stop[1] is None:
            x_start_stop[1] = w
        if y_start_stop[0] is None:
            y_start_stop[0] = 0
        if y_start_stop[1] is None:
            y_start_stop[1] = h
        # Compute the span of the region to be searched
        xspan = x_start_stop[1] - x_start_stop[0]
        yspan = y_start_stop[1] - y_start_stop[0]
        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
        # Compute the number of windows in x/y
        nx_buffer = np.int(xy_window[0] * (xy_overlap[0]))
        ny_buffer = np.int(xy_window[1] * (xy_overlap[1]))
        nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)
        ny_windows = np.int((yspan - ny_buffer) / ny_pix_per_step)
        # Initialize a list to append window positions to
        window_list = []
        # Loop through finding x and y window positions
        # Note: you could vectorize this step, but in practice
        # you'll be considering windows one by one with your
        # classifier, so looping makes sense
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                startx = xs * nx_pix_per_step + x_start_stop[0]
                endx = startx + xy_window[0]
                starty = ys * ny_pix_per_step + y_start_stop[0]
                endy = starty + xy_window[1]
                # Append window position to list
                window_list.append(np.array([startx, starty, endx, endy]))
        # Return the list of windows
        return window_list

    # Get scaled image + list of images for it
    # It is faster to have constant window size and different image sizes, then other way around
    def windows(self, image):
        windows = []
        for scale, generated_windows in zip(self.scales, self.generated_windows):
            # Resize image
            scaled_image = resize((image / 255.).astype(np.float64), (int(self.h * scale), int(self.w * scale), self.d), preserve_range=True).astype(np.float32) #imresize(image, (int(self.h * scale), int(self.w * scale)))
            windows.append((scaled_image, scale, generated_windows))

        return windows

    def draw(self, image, colors=[(0, 255, 0), (0, 100, 255), (0, 100, 100), (0, 255, 100)], width=2):
        for scale, color, generated_windows in zip(self.scales, colors, self.generated_windows):
            for window in generated_windows:
                window = (window / scale).astype(np.int)
                cv2.rectangle(image, (window[0], window[1]), (window[2], window[3]), color, width)
        return image
