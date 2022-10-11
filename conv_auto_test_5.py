import numpy as np
import matplotlib.pyplot as plt
import random
import keras

from tensorflow.keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,UpSampling2D
from keras.models import Model
from keras.datasets import mnist
from keras.optimizers import RMSprop

from PIL import Image

(train_X,train_Y), (test_X,test_Y) = mnist.load_data()



train_X = train_X.astype('float32') / 255.
test_X = test_X.astype('float32') / 255.
train_X = np.reshape(train_X, (len(train_X), 28, 28, 1))
test_X = np.reshape(test_X, (len(test_X), 28, 28, 1))


input_img = keras.Input(shape=(28, 28, 1))

model = Sequential()
model.add(Conv2D(64,(3, 3), activation='relu', padding='same', input_shape=(28,28,1)))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(32,(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(16,(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))

model.add(Conv2D(16,(3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32,(3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64,(3, 3), activation='relu'))
model.add(UpSampling2D((2, 2)))

model.add(Conv2D(1,(3,3), activation='sigmoid', padding='same'))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
model.summary()

history = model.fit(train_X, train_X, batch_size=128, epochs=10, validation_data=(test_X,test_X))

epochs=10
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss)+1)
plt.figure()
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()