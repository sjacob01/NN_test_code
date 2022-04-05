import numpy as np
import pandas as pd
from numpy import exp
from keras.datasets import mnist
from keras.utils import to_categorical
from matplotlib import pyplot

#### Define Activation Functions
def relu(x):
    return np.maximum(0,x)
    
    
# # Numerically stable softmax
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x/e_x.sum(axis=0)

 
###Define Network Structure and create weights, biases


def init_param():
                        #rows, cols
    W1 = np.random.randn(392,784)
    b1 = np.random.randn(392,1)
    W2 = np.random.randn(64,392)
    b2 = np.random.randn(64,1)
    W3 = np.random.rand(10,64)
    b3 = np.random.rand(10,1)
    return W1,b1,W2,b2,W3,b3


#Load mnist data into variables
(train_X, train_Y), (test_X, test_Y) = mnist.load_data()

#Display basic information about the data and data structure
print('Training data shape: ', train_X.shape, train_Y.shape)
print('Testing data shape: ', test_X.shape, test_Y.shape)

scale_data = train_X/255

# Resize the matrix from 28,28 to 784,1
train_X_formatted = scale_data.reshape(len(scale_data),784,1)

learning_rate = 0.01

def forward_prop_step(X,W1,b1,W2,b2,W3,b3):
   
    Z1 = W1.dot(X)+b1
    A1 = relu(Z1)
    Z2 = W2.dot(A1)+b2
    A2 = relu(Z2)   
    Z3 = W3.dot(A2)+b3
    A3 = softmax(Z3)
    
    return Z1,A1,Z2,A2,Z3,A3


def relu_derivative(x):
    return x>0
    

def softmax_derivative(x):
    # Numerically stable with large exponentials
    exps = np.exp(x - x.max())  
    return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
  


def back_prop_step(Z1,A1,Z2,A2,Z3,A3,W3,W2,X,Y):
    one_hot=to_categorical(Y, num_classes=10)
    dZ3 = A3 - one_hot.reshape(10,1)   
    dW3 = (1/A2.size) * dZ3.dot(A2.T)
    db3 = (1/A2.size) *np.sum(dZ3,1)   
    
    dZ2 = W3.T.dot(dZ3)*relu_derivative(Z2)    
    dW2 = (1/A1.size) * dZ2.dot(A1.T)
    db2 = (1/A1.size) * np.sum(dZ2,1)
   
    dZ1 = W2.T.dot(dZ2) * relu_derivative(Z1)
    dW1 = (1/X.size) * dZ1.dot(X.T)
    db1 = (1/X.size) * np.sum(dZ1,1)
    
    return dW1,db1,dW2,db2,dW3,db3


        # # Calculate W3 update
        # error = 2 * (output - y_train) / output.shape[0] * self.softmax(params['Z3'], derivative=True)
        # change_w['W3'] = np.outer(error, params['A2'])

        


def update_params(W1,b1,W2,b2,W3,b3,dW1,db1,dW2,db2,dW3,db3,learning_rate):
    
    W1 = W1 - learning_rate * dW1
    db1= db1.reshape(len(db1),1)
    b1 = b1 - learning_rate * db1      
    W2 = W2 - learning_rate * dW2
    db2= db2.reshape(len(db2),1)
    b2 = b2 - learning_rate * db2
    W3 = W3 - learning_rate * dW3
    db3= db3.reshape(len(db3),1)
    b3 = b3 - learning_rate * db3
    return W1,b1,W2,b2,W3,b3

 


def gradient_descent (X,Y, iterations, learning_rate):
    np.random.seed(100)
    W1,b1,W2,b2,W3,b3 = init_param()
    num_correct = 0
    n=0
   
    for i in range(iterations):
        Z1,A1,Z2,A2,Z3,A3 = forward_prop_step(X[i],W1,b1,W2,b2,W3,b3)
        dW1,db1,dW2,db2,dW3,db3 = back_prop_step(Z1,A1,Z2,A2,Z3,A3,W3,W2,X[i],Y[i])
        W1,b1,W2,b2,W3,b3 = update_params(W1,b1,W2,b2,W3,b3,dW1,db1,dW2,db2,dW3,db3,learning_rate)
        prediction = np.where(np.rint(A3)==1)
        n+=1
        if prediction[0] == Y[i]:          
            num_correct+=1
        if (i%1000 ==0):
           acc = num_correct/(n)*100           
           print("Accuracy: " + str(acc))
    print("Accuracy: " + str((num_correct/iterations) *100) + "%")
    return W1,b1,W2,b2,W3,b3

W1,b1,W2,b2,W3,b3 =gradient_descent(train_X_formatted,train_Y, 10000, learning_rate)


######################################################################

# Add Activation Plot code

