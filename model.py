import os
import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import numpy as np
import sklearn

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        if line[0] != 'center':
            samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

step_size = 6

def generator(gen_samples, batch_size=32):
    batch_size = batch_size // step_size
    correction = 0.2
    num_samples = len(gen_samples)
    while 1: # Loop forever so the generator never terminates
        S = sklearn.utils.shuffle(gen_samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = S[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                steering_center = float(batch_sample[3])
                steering_left = steering_center + correction
                steering_right = steering_center - correction
                steering_center_aug = steering_center * -1.0
                steering_left_aug = steering_left * -1.0
                steering_right_aug = steering_right * -1.0

                directory = './data/IMG/'
                filename_center = directory + batch_sample[0].split('/')[-1]
                filename_left = directory + batch_sample[1].split('/')[-1]
                filename_right = directory + batch_sample[2].split('/')[-1]
                image_center = cv2.imread(filename_center)
                image_left = cv2.imread(filename_left)
                image_right = cv2.imread(filename_right)
                image_center_aug = cv2.flip(image_center, 1)
                image_left_aug = cv2.flip(image_left, 1)
                image_right_aug = cv2.flip(image_right, 1)

                images.extend([ image_center,
                                image_left,
                                image_right,
                                image_center_aug,
                                image_left_aug,
                                image_right_aug
                                ])
                angles.extend([ steering_center,
                                steering_left,
                                steering_right,
                                steering_center_aug,
                                steering_left_aug,
                                steering_right_aug
                                ])

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# MODEL
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import EarlyStopping

model = Sequential()

model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation = 'relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation = 'relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation = 'relu'))
model.add(Dropout(0.2))
model.add(Convolution2D(64,3,3, activation = 'relu'))
model.add(Convolution2D(64,3,3, activation = 'relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model.fit_generator(train_generator,
                    samples_per_epoch= len(train_samples)*step_size,
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples)*step_size,
                    callbacks=[early_stopping],
                    #verbose = 1,
                    nb_epoch=200)

model.save('model.h5')
