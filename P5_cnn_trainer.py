from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils

from src.utils import *
from sklearn.model_selection import train_test_split

def model():
    input_img = Input(shape=(64, 64, 3))

    conv0 = Convolution2D(3, 1, 1, input_shape=(64, 64, 3), border_mode='valid', init='he_normal', activation='elu')(input_img)

    conv1 = Convolution2D(25, 5, 5, border_mode='same', subsample=(1, 1), activation='elu')(conv0)
    max_pool1 = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(conv1)
    drop1 = Dropout(0.1)(max_pool1)

    conv2 = Convolution2D(50, 5, 5, border_mode='same', subsample=(1, 1), activation='elu')(drop1)
    max_pool2 = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(conv2)
    drop2 = Dropout(0.2)(max_pool2)

    conv3 = Convolution2D(100, 5, 5, border_mode='same', subsample=(1, 1), activation='elu')(drop2)
    max_pool3 = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(conv3)
    drop3 = Dropout(0.3)(max_pool3)


    max_pool1_ = MaxPooling2D((2, 2), strides=(4, 4), border_mode='same')(drop1)
    max_pool2_ = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(drop2)

    merged = merge([max_pool1_, max_pool2_, drop3], mode='concat', concat_axis=3)
    flatten = Flatten()(merged)

    dense1 = Dense(500, activation='elu')(flatten)
    dense1 = Dropout(.5)(dense1)

    dense2 = Dense(200, activation='elu')(dense1)
    dense2 = Dropout(.5)(dense2)

    out = Dense(2, activation='softmax')(dense2)

    model = Model(input_img, out)
    return model

def model2():
    model = Sequential()
    model.add(Convolution2D(3, 1, 1, input_shape=(64, 64, 3), border_mode='valid', init='he_normal', activation='elu'))
    model.add(Convolution2D(32, 5, 5, border_mode='same', input_shape=(64, 64, 3), activation='elu'))
    model.add(Convolution2D(32, 5, 5, activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(64, 5, 5, border_mode='same', activation='elu'))
    model.add(Convolution2D(64, 5, 5, activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(128, 5, 5, border_mode='same', activation='elu'))
    model.add(Convolution2D(128, 5, 5, activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    return model

model_ = model2()
model_.summary()


#Load data
print('Load training data...')
cars = car_images()
notcars = not_car_images()
print('Training data info:')

print('Vehicles images:', len(cars))
print('Non-vehicles images:', len(notcars))

X = np.vstack((cars, notcars))
y = np.hstack((np.ones(len(cars)), np.zeros(len(notcars))))
y = np_utils.to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

datagen = ImageDataGenerator(featurewise_center=False,
                            featurewise_std_normalization=False,
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            zoom_range=0.2,
                            shear_range=0.1,
                            rotation_range=10.,)

datagen.fit(X_train)

model_.compile(optimizer=Adam(lr=1e-04), loss='categorical_crossentropy')#mean_squared_error')

nb_epoch = 30
batch_size = 256

import keras.callbacks
from keras.callbacks import LearningRateScheduler, ModelCheckpoint


class WeightsLogger(keras.callbacks.Callback):
    """
    Keeps track of model weights by saving them at the end of each epoch.
    """

    def __init__(self):
        super(WeightsLogger, self).__init__()

    def on_epoch_end(self, epoch, logs={}):
        self.model.save('model_epoch_{}.h5'.format(epoch + 1))


model_.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                            samples_per_epoch=X_train.shape[0],
                            nb_epoch=nb_epoch,
                            validation_data=(X_test, y_test),
                            callbacks=[ModelCheckpoint('model.h5',save_best_only=True)]
                           )

