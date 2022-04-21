import pandas as pd
import tensorflow as tf
import numpy as np
import math
import umap
import csv
import xlsxwriter
import umap.plot
import utils
from matplotlib import pyplot
from numpy import exp
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score
from keract import get_activations, display_activations

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


activations = get_activations(model, X_train[1:2], auto_compile=True)
[print(k, '->', v.shape, '- Numpy array') for (k, v) in activations.items()]
#print(activations)

classes = np.unique(Y_train)
print(classes)

#####################################

#Step 1
index_counter=[]

for a in range(10):   
    #In the mnsist test set find the indices where each distinct number exists
    column=[]    
    for b in range(len(Y_test)):
        if Y_test[b]==a:
            column.append(b)
    index_counter.append(column)


for a in range(10):
    # Create the csvs to hold the activations of each layer for each number
    var1 = 'class_'
    var2 = str(a)
    

   
    for b in range(1,6):
        var3 = '_layer_'
        var4 = str(b)
        var5 = '.csv'
        var6 = var1+var2+var3+var4+var5
        
        h = 'hidden_'+ str(b)
          
        for c in range(len(index_counter[a])):
            
            image=X_test[index_counter[a][c:c+1]]
            activations = get_activations(model,image)
            layer_activations = activations[h].T
            with open(var6,'a+',newline='') as f:
                 writer = csv.writer(f)    
                 writer.writerow(layer_activations)
  