# This script creates a basic neural network model using tensorflow data sets
# June 2022

import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import math
import pymsgbox
import os
import umap
import umap.plot
import utils
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from matplotlib import pyplot
from numpy import exp
from tensorflow.keras.datasets import *
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score
from keract import get_activations, display_activations, display_heatmaps


## ----------------------------------------------------------------------------------------------
# This function allows the user to select the data used to create the neural network model
def select_dataset_to_model(options):
    print("\n")
    print("Please select the data set to model:")

    for idx, element in enumerate(options):
        print("{}) {}".format(idx + 1, element))
    
    i = input("Enter number: ")
    try:
        if 0 < int(i) <= len(options):
            return int(i) - 1
        else: 
            print('Select a number between 1 and ' + str(len(options)))
            return select_dataset_to_model(options)
            
    except:
        pass
    return None


dataset_to_model = select_dataset_to_model(tfds.list_builders())


## ------------------------------------------------------------------------------------------------
# This section normalizes the data and trains the model

tf.random.set_seed(10)

(Model_training_data, Model_training_data_labels), (Model_test_data, Model_test_data_labels) = eval(tfds.list_builders()[dataset_to_model]+'.load_data()')

Model_training_data = tf.keras.utils.normalize(Model_training_data,axis=1)
Model_test_data = tf.keras.utils.normalize(Model_test_data,axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(588, activation=tf.nn.relu,name="hidden_1"))
model.add(tf.keras.layers.Dense(392, activation=tf.nn.relu,name="hidden_2"))
model.add(tf.keras.layers.Dense(196, activation=tf.nn.relu,name="hidden_3"))
model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu,name="hidden_4"))
model.add(tf.keras.layers.Dense(32, activation=tf.nn.relu,name="hidden_5"))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax,name="hidden_6"))

model.compile(optimizer='sgd',loss='sparse_categorical_crossentropy',metrics='accuracy')


#Train the model on the desired data set
def fit_model(number_of_epochs):
   
    model.fit(Model_training_data,Model_training_data_labels,epochs=number_of_epochs)
    score = model.evaluate(Model_training_data,Model_training_data_labels)
        
    if score[1] < .98:
        number_of_epochs += 10
        return fit_model(number_of_epochs)
 
number_of_epochs = 10 
fit_model(number_of_epochs)

##----------------------------------------------------------------------------------------------
# This function enables the user to enter the path where the model is saved to
def set_path_to_save_model():
    print("\n")
    print("Save model to current working directory?")
    i = input("Enter yes/no: ")
    try:
        if i.lower() == 'yes':
            # Get the current working directory
            cwd = os.getcwd()

            model.save(cwd+'/Current_model')
            print('Model saved in the "Current_model" folder')
            return          
       
        else: 
            path = input('Enter the path to save the model: ')
            model.save(path)
            return 
    except:
        pass
    return None

set_path_to_save_model()

# Reference: https://programmerah.com/python-error-certificate-verify-failed-certificate-has-expired-40374/
