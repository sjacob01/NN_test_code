import pandas as pd
import tensorflow as tf
import numpy as np
import math
import os
import umap
import umap.plot
import utils
from matplotlib import pyplot
from numpy import exp
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score
from keract import get_activations, display_activations, display_heatmaps

tf.random.set_seed(10)

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
(F_train, G_train), (F_test, G_test) = fashion_mnist.load_data()


X_train = tf.keras.utils.normalize(X_train,axis=1)
X_test = tf.keras.utils.normalize(X_test,axis=1)

F_train = tf.keras.utils.normalize(F_train,axis=1)
F_test = tf.keras.utils.normalize(F_test,axis=1)


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(588, activation=tf.nn.relu,name="hidden_1"))
model.add(tf.keras.layers.Dense(392, activation=tf.nn.relu,name="hidden_2"))
model.add(tf.keras.layers.Dense(196, activation=tf.nn.relu,name="hidden_3"))
model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu,name="hidden_4"))
model.add(tf.keras.layers.Dense(32, activation=tf.nn.relu,name="hidden_5"))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax,name="hidden_6"))

model.compile(optimizer='sgd',loss='sparse_categorical_crossentropy',metrics='accuracy')


#Train the model on the mnist data set
model.fit(F_train,G_train,epochs=60)

#Save the neural network model

path = os.path.join("C:/Users/me/Documents/Python Scripts/NN_activation_results/saved_fashion_model")
model.save('path/my_fashion_model')

activations = get_activations(model, F_train[1:2], auto_compile=True)
[print(k, '->', v.shape, '- Numpy array') for (k, v) in activations.items()]

classes = np.unique(Y_train)
print('mnist classes: ', classes)

F_classes = np.unique(G_train)
print('fashion_mnist classes: ', F_classes)


#Step 1
index_counter=[]
labels_counter=[]


for a in range(len(classes)):   
    #In the mnsist test set find the indices where each distinct number exists
    column=[]    
    label_column=[]
    
    for b in range(len(Y_test)):
        if Y_test[b]==a:
            column.append(b)
            label_column.append(a)
            
    index_counter.append(column)
    labels_counter.append(label_column)


for a in range(len(F_classes)):   
    #In the fashion_mnsist test set find the indices where each distinct number exists
    column=[]    
    label_column=[]
    
    for b in range(len(G_test)):
        if G_test[b]==a:
            column.append(b)
            label_column.append(a+10)
            
    index_counter.append(column)
    labels_counter.append(label_column)



#Step 2

for c in range(1,6):
     
    nodes = model.layers[c].output_shape[1]
    activation_col_names=[]
    var4 = 'df'
    var5 = str(c)
    var6 = var4+var5
    
    h = 'hidden_'+ str(c)
    #Never grow a dataframe
    data =[]
    
    for d in range(nodes):
        var1 = 'node_'
        var2 = str(d)
        var3 = var1+var2
        activation_col_names.append(var3)
    
    for e in range(len(X_test)):
        
        image=X_test[e:e+1]
        activations = get_activations(model,image)
        layer_activations = activations[h].T
        layer_activations = np.reshape(layer_activations,nodes)
        data.append(layer_activations)
        
    for e in range(len(F_test)):
        
        image=F_test[e:e+1]
        activations = get_activations(model,image)
        layer_activations = activations[h].T
        layer_activations = np.reshape(layer_activations,nodes)
        data.append(layer_activations)
    
    var6 = pd.DataFrame(columns=activation_col_names)   
    
    var6 = pd.DataFrame(data)
    
 
    v= index_counter[0]+index_counter[1]+index_counter[2]+index_counter[3]+index_counter[4]+index_counter[5]+index_counter[6]+index_counter[7]+index_counter[8]+index_counter[9]\
    +index_counter[10]+index_counter[11]+index_counter[12]+index_counter[13]+index_counter[14]+index_counter[15]+index_counter[16]+index_counter[17]+index_counter[18]+index_counter[19]
    
    v_label=labels_counter[0]+labels_counter[1]+labels_counter[2]+labels_counter[3]+labels_counter[4]+labels_counter[5]+labels_counter[6]+labels_counter[7]+labels_counter[8]+labels_counter[9]\
    +labels_counter[10]+labels_counter[11]+labels_counter[12]+labels_counter[13]+labels_counter[14]+labels_counter[15]+labels_counter[16]+labels_counter[17]+labels_counter[18]+labels_counter[19]
    
  
   
    np.save(
        file="C:/Users/me/Documents/Python Scripts/NN_activation_results/layer_" +str(c)+"_result_labels.npy",
        arr=np.array(v_label),
        allow_pickle=False,
        fix_imports=False,
        )
        
    np.save(
        file="C:/Users/me/Documents/Python Scripts/NN_activation_results/layer_" +str(c)+"_results.npy",
        arr=var6,
        allow_pickle=False,
        fix_imports=False,
        )