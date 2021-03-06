# This script creates a basic neural network model using tensorflow data sets
# June 2022
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




# This function allows the user to select the data to be used as the adversarial input
def select_adversarial_dataset(options, model_choice):
    print("\n")
    print("Please select the adversarial input data:")

    for idx, element in enumerate(options):
        print("{}) {}".format(idx + 1, element))
    
    i = input("Enter number: ")
       
    try:
        if (model_choice+1) == int(i):
            
            pymsgbox.alert('Choose another data set. Current selection used to create model.', 'Title')
            return select_adversarial_dataset(options, model_choice)            
        elif 0 < int(i) <= len(options):
            return int(i) - 1
        else: 
            print('Select a number between 1 and ' + str(len(options)))
            return select_adversarial_dataset(options, model_choice)
    except:
        pass
    return None




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


def class_indices(index_counter,labels_counter,classes,labels, bias):
# This function identifies the indices where each class exists in the data set    
     for a in range(len(classes)):   
            column=[]    
            label_column=[]
            
            for b in range(len(labels)):
                if labels[b]==a:
                    column.append(b)
                    label_column.append(a+bias)
                    
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



def set_path_to_save_file():
# This function enables the user to set the path of where to save the activations
    print('Do you want to save the file to the current working directory?')
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




def capture_activations(nn_layers,model,Model_test_data,Adversarial_test_data,index_counter, labels_counter,total_number_of_classes):

    for a in range(1,nn_layers):
        nodes = model.layers[a].output_shape[1]
        activation_col_names=[]
        var4 = 'df'
        var5 = str(a)
        var6 = var4+var5
        
        h = 'hidden_'+ str(a)
        #Never grow a dataframe
        data =[]
        
        for b in range(nodes):
            var1 = 'node_'
            var2 = str(b)
            var3 = var1+var2
            activation_col_names.append(var3)
        
        data = capture_image_activations(Model_test_data, model, data, nodes,h)
        data = capture_image_activations(Adversarial_test_data, model, data, nodes,h)
       
        var6 = pd.DataFrame(columns=activation_col_names)   
    
        var6 = pd.DataFrame(data)
        #print(var6)     
           
        arrays_of_class_indices=[]               
        arrays_of_class_indices = cat_array_data(arrays_of_class_indices, index_counter, total_number_of_classes)
      
        array_of_class_labels=[]        
        array_of_class_labels=cat_array_data(array_of_class_labels, labels_counter, total_number_of_classes)
               
        path = set_path_to_save_file()
       
        arr=np.array(array_of_class_labels)
                 
        np.save(
            file=path+ "/layer_" +str(a)+"_result_labels.npy",
            arr=np.array(array_of_class_labels),
            allow_pickle=False,
            fix_imports=False,
            )
            
        np.save(
            file=path + "/layer_" +str(a)+"_results.npy",
            arr=var6,
            allow_pickle=False,
            fix_imports=False,
            )
       



def main():
    
    #Select the data set to model
    dataset_to_model = select_dataset_to_model(tfds.list_builders())
    
    #Select adversarial data set
    adversarial_input_data = select_adversarial_dataset(tfds.list_builders(), dataset_to_model)

    
    # This section normalizes the data and trains the model
    tf.random.set_seed(10)
    
    (Model_training_data, Model_training_data_labels), (Model_test_data, Model_test_data_labels) = eval(tfds.list_builders()[dataset_to_model]+'.load_data()')
    (Adversarial_training_data, Adversarial_training_data_labels), (Adversarial_test_data, Adversarial_test_data_labels) = eval(tfds.list_builders()[adversarial_input_data] +'.load_data()')

    Model_training_data = tf.keras.utils.normalize(Model_training_data,axis=1)
    Model_test_data = tf.keras.utils.normalize(Model_test_data,axis=1)
        
    Adversarial_training_data = tf.keras.utils.normalize(Adversarial_training_data,axis=1)
    Adversarial_test_data = tf.keras.utils.normalize(Adversarial_test_data,axis=1)


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
    set_path_to_save_model(model)
   
    
    Model_classes = np.unique(Model_training_data_labels)
    print('model classes: ', Model_classes)
    length_input_1 = len(Model_classes)
   
    Adversarial_input_classes = np.unique(Adversarial_training_data_labels)
    print('adversarial input classes: ', Adversarial_input_classes)
    
    length_input_2 = len(Adversarial_input_classes)
    total_number_of_classes = length_input_1+length_input_2
     
    index_counter=[]
    labels_counter=[]
    
    bias =0 
    [index_counter ,labels_counter] = class_indices(index_counter,labels_counter, Model_classes, Model_test_data_labels,bias)
   
    bias =length_input_2
    [index_counter, labels_counter] = class_indices(index_counter,labels_counter,Adversarial_input_classes, Adversarial_test_data_labels,bias)
   
    capture_activations(nn_layers,model,Model_test_data,Adversarial_test_data,index_counter, labels_counter,total_number_of_classes)


    

if __name__ == "__main__":
    main()




# Reference: https://programmerah.com/python-error-certificate-verify-failed-certificate-has-expired-40374/
