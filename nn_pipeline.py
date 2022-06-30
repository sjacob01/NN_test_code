# This script creates a basic neural network model using tensorflow data sets
# June 2022

import tensorflow as tf
import tensorflow_datasets as tfds
import os
import utils
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




def set_path_to_save_model():
# This function enables the user to enter the path where the model is saved to
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
            
     return index_counter labels_counter

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
    set_path_to_save_model()
    Model_classes = np.unique(Model_training_data_labels))
    print('model classes: ', classes)

    Adversarial_input_classes = np.unique(Adversarial_training_data_labels)
    print('adversarial input classes: ', Adversarial_input_classes)


    #Step 1
    index_counter=[]
    labels_counter=[]
    
    [index_counter labels_counter] = class_indices(index_counter,labels_counter,Model_test_data, Model_test_data_labels)
    
    [index_counter labels_counter] = class_indices(index_counter,labels_counter,Adversarial_test_data, Adversarial_test_data_labels)
    


    

if __name__ == "__main__":
    main()




# Reference: https://programmerah.com/python-error-certificate-verify-failed-certificate-has-expired-40374/
