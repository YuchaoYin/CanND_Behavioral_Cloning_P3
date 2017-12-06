from os.path import join
import cv2
import numpy as np
from sklearn.utils import shuffle
import random

def data_process(data, batch_size, path, correction):
    X = np.zeros(shape=(batch_size, 160, 320, 3), dtype=np.float32)
    y = np.zeros(shape=(batch_size,), dtype=np.float32)

    data_shuffle = shuffle(data)
    counter = 0
    while counter < batch_size:
        center, left, right, steering, throttle, brake, speed = data_shuffle.pop()
        if len(center) > 40:
            image_path = ''
        else:
            image_path = path

        steering = float(steering)

        camera = random.choice(['center','left','right'])
        if camera == 'center':
            img = cv2.imread(join(image_path, center.strip()))
            steering = steering
        elif camera == 'left':
            img = cv2.imread(join(image_path, left.strip()))
            steering = steering + correction
        elif camera == 'right':
            img = cv2.imread(join(image_path, right.strip()))
            steering = steering - correction

        if random.choice([True, False]):
            img = cv2.flip(img, 1)
            steering *= -1.0

        img = cv2.cvtColor(img, code=cv2.COLOR_BGR2RGB)

        X[counter] = img
        y[counter] = steering
        counter += 1

    return X, y

def generator(data, batch_size, path, correction):
    while True:
        X, y = data_process(data, batch_size, path, correction)
        yield X, y