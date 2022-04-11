import pandas as pd
import tensorflow as tf
import numpy as np
import math
import umap
import umap.plot
from matplotlib import pyplot
from numpy import exp
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score

tf.random.set_seed(10)
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
x_train=X_train


X_train = tf.keras.utils.normalize(X_train,axis=1)
X_test = tf.keras.utils.normalize(X_test,axis=1)


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(392, activation=tf.nn.relu,name="hidden_1"))
model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu,name="hidden_2"))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='sgd',loss='sparse_categorical_crossentropy',metrics='accuracy')
#model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics='accuracy')

# model.compile(loss='binary_crossentropy', optimizer='sgd', metrics='accuracy')

model.fit(X_train,Y_train,epochs=10)

#####################################


scale_data = x_train/255

# Resize the matrix from 28,28 to 784,1
X_train_formatted = scale_data.reshape(len(scale_data),784,1)


#print(len(model.layers))
h1= model.layers[1]
h2 = model.layers[2]

w1,b1 = h1.get_weights()
w1=w1.reshape(392,784)
b1=b1.reshape(392,1)


w2,b2 = h2.get_weights()

###############################


#### Define Activation Functions
def relu(x):
    return np.maximum(0,x)


activation_col_names=[]
for x in range(w1.shape[0]):
    var1 = 'pixels_'
    var2 = str(x)
    var3= var1+var2
    activation_col_names.append(var3)


df=pd.DataFrame(columns=activation_col_names)


index_counter=[]
labels_counter=[]
activations=[]
#In the mnsist test set find the indices where each distinct number exists
for a in range(10):
    column=[]
    label_column=[]
      
    for b in range(len(Y_train)):
        
        if Y_train[b]==a:           
            column.append(b)
            label_column.append(a)
                           
    index_counter.append(column)
    labels_counter.append(label_column)

print("section_3")
print(X_train_formatted.shape)


for b in range(X_train_formatted.shape[0]):
    print(b)
    Z1 = w1.dot(X_train_formatted[b])    
    A1 = relu(Z1)    
    df.loc[b,:]=A1.T


#W1.dot(new_X[b:(b+1)].T)+b1
print("section_4")



# v= index_counter[1]+index_counter[2]+index_counter[3]+index_counter[4]+index_counter[5]+index_counter[6]+index_counter[7]+index_counter[8]+index_counter[9]
# v_label=labels_counter[1]+labels_counter[2]+labels_counter[3]+labels_counter[4]+labels_counter[5]+labels_counter[6]+labels_counter[7]+labels_counter[8]+labels_counter[9]
# arr=np.array(v_label)



# mapper = umap.UMAP().fit(df.iloc[v])
# p=umap.plot.points(mapper, labels=arr,color_key_cmap='Paired', background='black')
# umap.plot.plt.show()