import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import os
import utils
import math
import umap
import umap.plot
from matplotlib import pyplot
from numpy import exp
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from tensorflow.keras.datasets import *
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score
from keract import get_activations, display_activations, display_heatmaps
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


## ----------------------------------------------------------------------------------------------

def select_dataset_to_model(options):
# This function allows the user to select the data used to create the neural network model
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



def fit_model(number_of_epochs, model,Model_training_data,Model_training_data_labels):
#Train the model on the desired data set
   
    model.fit(Model_training_data,Model_training_data_labels,epochs=number_of_epochs)
    score = model.evaluate(Model_training_data,Model_training_data_labels)
        
    if score[1] < .98:
        number_of_epochs += 10
        return fit_model(number_of_epochs)



def class_indices(index_counter,labels_counter,classes,labels):
# This function identifies the indices where each class exists in the data set    
     for a in range(len(classes)):   
            column=[]    
            label_column=[]
            
            for b in range(len(labels)):
                if labels[b]==a:
                    column.append(b)
                    label_column.append(a)
                    
            index_counter.append(column)
            labels_counter.append(label_column)
            
     return index_counter, labels_counter
     


def capture_image_activations(input_data,model,data, nodes,h):
        for e in range(len(input_data)):
        
            image=input_data[e:e+1]
            activations = get_activations(model,image)           
            layer_activations = activations[h].T
            layer_activations = np.reshape(layer_activations,nodes)
            data.append(layer_activations)
        return data 
        

def cat_array_data(arrays_of_class_data, input_data, counter): 
    arrays_of_class_data = input_data[0]
    
    for a in range(1,counter): 
        arrays_of_class_data = arrays_of_class_data  + input_data[a]
    return arrays_of_class_data 


def set_path_to_save_model(model):
# This function enables the user to enter the path where the model is saved to
    print("\n")
    print("Save model to current working directory?")
    i = input("Enter yes/no: ")
    try:
        if i.lower() == 'yes':
            # Get the current working directory
            cwd = os.getcwd()
            print(cwd)
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

def set_path_to_save_file():
# This function enables the user to set the path of where to save the activations
    print('This file will be saved. Do you want to save the file to the current working directory?')
    i = input("Enter yes/no: ")
    try:
        if i.lower() == 'yes':
            # Get the current working directory
            cwd = os.getcwd()
       
            return cwd
       
        else: 
            path = input('Enter the path to save the file: ')
            
            return path
    except:
        pass
    return None




def capture_activations(nn_layers,model,Model_test_data,index_counter, labels_counter,total_number_of_classes):
    
    for a in range(total_number_of_classes):
        # Create the csvs to hold the activations of each layer for each number
        var1 = 'class_'
        var2 = str(a)
        
        for b in range(1,nn_layers+1):
            var3 = '_layer_'
            var4 = str(b)
            var5 = '.csv'
            var6 = var1+var2+var3+var4+var5
            
                      
            h = 'hidden_'+ str(b)
            length_class=len(index_counter[a])
        
            nodes = model.layers[b].output_shape[1]
      
            column_names=[]
            data = []
            
            for c in range(nodes):
                var7 = 'node_'
                var8 = str(c)
                var9 = var7+var8
                column_names.append(var9)
                          
       
            df=pd.DataFrame(columns = column_names) 
            
            for d in range(length_class):
                      
                image=Model_test_data[index_counter[a][d:d+1]]
                activations = get_activations(model,image)
                layer_activations = activations[h]
                layer_activations = layer_activations.reshape(nodes)
                df.loc[d] = layer_activations
                
                data.append(str(index_counter[a][d]))
          
            df.insert(0,'Image_Index',data,True)  
          
            
            df.to_csv(var6, index=False)
  
  


def main():
    
    #Select the data set to model
    dataset_to_model = select_dataset_to_model(tfds.list_builders())


    # This section normalizes the data and trains the model
    tf.random.set_seed(10)

    (Model_training_data, Model_training_data_labels), (Model_test_data, Model_test_data_labels) = eval(tfds.list_builders()[dataset_to_model]+'.load_data()')

    Model_training_data = tf.keras.utils.normalize(Model_training_data,axis=1)
    Model_test_data = tf.keras.utils.normalize(Model_test_data,axis=1)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())

    nn_layers = int(input("Enter the number of layers desired: "))

    index = 0
    while index < nn_layers:
        nodes = input("Enter the desired number of nodes for layer_" + str(index+1) +": ")

        layer_name = "hidden_"+str(index+1)
        model.add(tf.keras.layers.Dense(str(nodes), activation=tf.nn.relu,name=layer_name))
        index +=1
    else:
        layer_name = "hidden_"+str(index+1)
    
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax,name=layer_name))

    model.compile(optimizer='sgd',loss='sparse_categorical_crossentropy',metrics='accuracy')

    number_of_epochs = 10 
    fit_model(number_of_epochs, model,Model_training_data,Model_training_data_labels)

    #Save the model
    #set_path_to_save_model(model)


    Model_classes = np.unique(Model_training_data_labels)
    print('model classes: ', Model_classes)
    total_number_of_classes = len(Model_classes)
    
    index_counter=[]
    labels_counter=[]
    
  
    [index_counter ,labels_counter] = class_indices(index_counter,labels_counter, Model_classes, Model_test_data_labels)
    
    capture_activations(nn_layers,model,Model_test_data,index_counter, labels_counter,total_number_of_classes)
   
  




if  __name__ == "__main__":
    main()


