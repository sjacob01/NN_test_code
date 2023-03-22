from tensorflow.keras.datasets import cifar10

import numpy as np
import pandas as pd

import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import utils


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D, GlobalAveragePooling2D, Activation, BatchNormalization, LeakyReLU
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from matplotlib import pyplot



def cnn_model():
    
    model = Sequential()
    
    # First Conv layer
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation=keras.layers.LeakyReLU(alpha=0.1), padding='same', kernel_regularizer=regularizers.l2(1e-4), input_shape=(32,32,3)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))
    
    model.add(Conv2D(filters=128, kernel_size=(3,3), activation=keras.layers.LeakyReLU(alpha=0.1), padding='same', kernel_regularizer=regularizers.l2(1e-4)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))
    
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation=keras.layers.LeakyReLU(alpha=0.1), padding='same', kernel_regularizer=regularizers.l2(1e-4)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))
    
    model.add(Conv2D(filters=128, kernel_size=(3,3), activation=keras.layers.LeakyReLU(alpha=0.1), padding='same', kernel_regularizer=regularizers.l2(1e-4)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))
    
    model.add(Conv2D(filters=256, kernel_size=(3,3), activation=keras.layers.LeakyReLU(alpha=0.1), padding='same', kernel_regularizer=regularizers.l2(1e-4)))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(Dropout(0.5))
    
    # Second Conv layer
    model.add(Conv2D(filters=128, kernel_size=(3,3), activation=keras.layers.LeakyReLU(alpha=0.1), padding='same', kernel_regularizer=regularizers.l2(1e-4)))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(Dropout(0.4))
    
    # Third, fourth, fifth convolution layer
    
    
    model.add(Conv2D(filters=256, kernel_size=(3,3), activation=keras.layers.LeakyReLU(alpha=0.1), padding='same', kernel_regularizer=regularizers.l2(1e-4)))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(Dropout(0.5))
    
   

    # Fully Connected layers
    model.add(Flatten())
    
  
    model.add(Dense(128, activation=keras.layers.LeakyReLU(alpha=0.1)))
    model.add(Dropout(0.3))
    
    model.add(Dense(10, activation='softmax'))
    
    model.summary()
    
    
    
    return model

def main():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    print('Training set shape:', X_train.shape)
    print('Test set shape:', X_test.shape)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    
    Model_classes = np.unique(y_train)
    print('model classes: ', Model_classes)
    num_classes = len(Model_classes)

    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2)
    print(X_train.shape)
    
    
    model = cnn_model()
    
        
    model.compile(loss='categorical_crossentropy',
             optimizer=Adam(learning_rate=0.0003, decay=1e-6),
             metrics=['accuracy'])
             
    history = model.fit(X_train, y_train, batch_size = 64,
                    steps_per_epoch = len(X_train) // 64, 
                    epochs = 50, 
                    validation_data= (X_valid, y_valid),
                    verbose=1)
                    
    pd.DataFrame(history.history).plot()
    
    scores = model.evaluate(X_test, y_test)
    
    pred = model.predict(X_test)
    
    #activations = get_activations(model,image)
    
    labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    y_pred = np.argmax(pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    errors = y_pred - y_true != 0
    
    print(classification_report(y_true, y_pred))
    
    fig, axes = pyplot.subplots(5, 5, figsize=(12,12))
    axes = axes.ravel()

    for i in np.arange(25):
        axes[i].imshow(X_test[i])
        axes[i].set_title('True: %s \nPredict: %s' % (labels[y_true[i]], labels[y_pred[i]]))
        axes[i].axis('off')
        pyplot.subplots_adjust(wspace=1)
    pyplot.show()
    
if  __name__ == "__main__":
    main()
