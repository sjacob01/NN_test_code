import keras
from matplotlib import pyplot as plt
import numpy as np

from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D,LeakyReLU
from keras.models import Model
from keras.datasets import mnist
from keras.optimizers import RMSprop

(train_X,train_Y), (test_X,test_Y) = mnist.load_data()



train_X = train_X.astype('float32') / 255.
test_X = test_X.astype('float32') / 255.
train_X = np.reshape(train_X, (len(train_X), 28, 28, 1))
test_X = np.reshape(test_X, (len(test_X), 28, 28, 1))


input_img = keras.Input(shape=(28, 28, 1))

x = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', name='conv_1')(input_img)
x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', name='conv_2')(x)
x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
x = Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same', name='conv_3')(x)
x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

x = Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same', name='conv_4')(x)
x = UpSampling2D(size=(2, 2))(x)
x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', name='conv_5')(x)
x = UpSampling2D(size=(2, 2))(x)
x = Conv2D(64, kernel_size=(3, 3), activation='relu', name='conv_6')(x)
x = UpSampling2D(size=(2, 2))(x)

output = Conv2D(filters=1, kernel_size=(3,3), padding ='same', activation='sigmoid', name='output')(x)

model = Model(input_img, output)
model.compile(optimizer='nadam', loss='binary_crossentropy',metrics=['accuracy'])

model.summary()

history = model.fit(train_X, train_X, batch_size=128, epochs=60, validation_data=(test_X,test_X))

out_images = model.predict(test_X)

epochs=60
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(epochs)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


n = 10

indices = np.random.randint(len(test_X), size=n)
images1 = test_X[indices, :]
images2 = out_images[indices, :]

plt.figure(figsize=(20, 4))
for i, (image1, image2) in enumerate(zip(images1, images2)):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(image1.reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(image2.reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()