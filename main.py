import csv
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from data_processing import generator
import matplotlib.pyplot as plt

lines = []
with open ('/home/yuchao/CarND-Behavioral-Cloning-P3/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

training_data, validation_data = train_test_split(lines, test_size=0.1, random_state=1)

batch_size = 128
path = '/home/yuchao/CarND-Behavioral-Cloning-P3/data/'
correction = 0.2

def resize(img):
    import tensorflow as tf
    return tf.image.resize_images(img, (80, 160))


# my model based on the paper
model = Sequential()
model.add(Lambda(lambda x:x/127.5 - 1.0, input_shape=(160,320,3)))
model.add(Lambda(resize))
model.add(Cropping2D(cropping=((20,10),(0,0))))

model.add(Convolution2D(24, 5, 5, border_mode='same', activation='elu'))
model.add(MaxPooling2D(2,2))

model.add(Convolution2D(36, 5, 5, border_mode='same', activation='elu'))
model.add(MaxPooling2D(2,2))

model.add(Convolution2D(48, 5, 5, border_mode='same', activation='elu'))
model.add(MaxPooling2D(2,2))

model.add(Convolution2D(64, 3, 3, border_mode='same', activation='elu'))
model.add(Convolution2D(64, 3, 3, border_mode='same', activation='elu'))

model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(1))

#model.load_weights('model.h5')
#adam = Adam(lr=0.0001)
model.compile(loss='mse', optimizer='adam')

checkpointer = ModelCheckpoint('/home/yuchao/CarND-Behavioral-Cloning-P3/data/weights.hdf5')

# training
history_object = model.fit_generator(generator=generator(training_data, batch_size, path, correction),
                    steps_per_epoch=300, epochs=8,
                    validation_data=generator(validation_data, batch_size, path, correction),
                    validation_steps=100, callbacks=[checkpointer])


model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model.h5")
model.save('model.h5')

# print the keys contained in the history object
print(history_object.history.keys())

# plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
