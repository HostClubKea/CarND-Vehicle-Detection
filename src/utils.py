import os
import glob
import pickle
import matplotlib.image as mpimg
import numpy as np

def get_images(dir):
    files = os.listdir(dir)
    images = []
    names = []
    for file in files:
        image = mpimg.imread(os.path.join(dir, file))
        images.append(image)
        names.append(file)
    return images, names


def car_images():
    cars = []
    cars_files = glob.glob('data/vehicles/*/*.png')
    for file in cars_files:
        cars.append(mpimg.imread(file))
    return cars #np.asarray(cars)

def not_car_images():
    notcars = []
    notcars_files = glob.glob('data/non-vehicles/*/*.png')
    for file in notcars_files:
        notcars.append(mpimg.imread(file))
    return np.asarray(notcars)

def save_pickle(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj
