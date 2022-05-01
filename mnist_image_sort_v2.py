import pandas as pd
import tensorflow as tf
import numpy as np
import math
import utils
import csv
from matplotlib import pyplot
from numpy import exp
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score
from keract import get_activations, display_activations, display_heatmaps
from keras import backend as K

tf.random.set_seed(10)
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
x_train=X_train


X_train = tf.keras.utils.normalize(X_train,axis=1)
X_test = tf.keras.utils.normalize(X_test,axis=1)


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(588, activation=tf.nn.relu,name="hidden_1"))
model.add(tf.keras.layers.Dense(392, activation=tf.nn.relu,name="hidden_2"))
model.add(tf.keras.layers.Dense(196, activation=tf.nn.relu,name="hidden_3"))
model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu,name="hidden_4"))
model.add(tf.keras.layers.Dense(32, activation=tf.nn.relu,name="hidden_5"))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax,name="hidden_6"))

model.compile(optimizer='sgd',loss='sparse_categorical_crossentropy',metrics='accuracy')
#model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics='accuracy')

# model.compile(loss='binary_crossentropy', optimizer='sgd', metrics='accuracy')

model.fit(X_train,Y_train,epochs=10)


classes = np.unique(Y_train)
print(classes)

#####################################

#Step 1
index_counter=[]
labels_counter=[]


for a in range(10):   
    #In the mnsist test set find the indices where each distinct number exists
    column=[]    
    label_column=[]
    
    for b in range(len(Y_test)):
        if Y_test[b]==a:
            column.append(b)
            label_column.append(a)
            
    index_counter.append(column)
    labels_counter.append(label_column)


# Step 2
# Get the predictions
predictions = model.predict(X_test)

#Step 3
# For each class, identify which image is misidentified by class

for b in range(0,10):
    var1 = 'misidentified_images_for_class_'
    var2 = str(b)
    var3 = '.csv'
    var4= var1+var2+var3
   
       
    misidentified_images=[]
    incorrect_result =[]
    for c in range(len(index_counter[b])):
    
        result = np.where(predictions[index_counter[b][c]] == np.amax(predictions[index_counter[b][c]]))
      
        if (result[0]!=b):
            misidentified_images.append(str(index_counter[b][c]))
            incorrect_result.append(str(result[0]))
        
     
    with open(var4,'a+',newline='') as f:
         writer = csv.writer(f)    
         writer.writerow(misidentified_images)
         writer.writerow(incorrect_result)  

