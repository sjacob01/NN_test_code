import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np
import math
import umap
import umap.plot
import utils
import seaborn as sns
import tensorflow_datasets as tfds
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from matplotlib import pyplot
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from keract import get_activations, display_activations, display_heatmaps
from tensorflow.keras.datasets import *
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
#importing Seaborn's to use the heatmap 
import seaborn as sns
import matplotlib as mpl
from sklearn.datasets import make_classification
from sklearn import preprocessing
from tensorflow.keras.losses import SparseCategoricalCrossentropy


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



def fit_model(number_of_epochs, model,model_training_data,model_training_data_labels):
#Train the model on the desired data set
    
  
    model.fit(model_training_data,model_training_data_labels,epochs=number_of_epochs)
    score = model.evaluate(model_training_data,model_training_data_labels)

    return model




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



def capture_activations(nn_layers, model, model_test_data, dataframe_dict,index_counter ,labels_counter):

    layer_activations = []
    layer_nodes=[]
    for c in range(1,nn_layers+1):

        L1 = 'layer_'
        L2 = str(c)
        L3 = L1+L2+'_new_Act_data_df.csv'

        nodes = model.layers[c].output_shape[1]
        activation_col_names=[]

        h = 'hidden_'+ str(c)
        #Never grow a dataframe
        data =[]

        for d in range(nodes):
            var1 = 'node_'
            var2 = str(d)
            var3 = var1+var2
            activation_col_names.append(var3)

        for e in range(len(model_test_data)):

            image=model_test_data[e:e+1]
            activations = get_activations(model,image)
            layer_activations = activations[h].T
            layer_activations = np.reshape(layer_activations,nodes)
            data.append(layer_activations)


        activation_dataframe = pd.DataFrame(data)

        activation_dataframe.columns = [activation_col_names]          
        
        #activation_dataframe.to_csv(L3, index=False)

        dataframe_dict[c]= activation_dataframe
        layer_nodes.append(nodes)

    return dataframe_dict, layer_nodes


def determine_pca_components(model_training_data, model_test_data, model_validation_data):
    #This function reduces the dimensionality of the data


    pca_784 = PCA(n_components=784)
    pca_784.fit(model_training_data)

    pyplot.grid()
    pyplot.plot(np.cumsum(pca_784.explained_variance_ratio_ * 100))
    pyplot.xlabel('Number of components')
    pyplot.ylabel('Explained variance')
    pyplot.show()

    pyplot.style.use("ggplot") 
    pyplot.plot(pca_784.explained_variance_, marker='o')
    pyplot.xlabel("Eigenvalue number")
    pyplot.ylabel("Eigenvalue size")
    pyplot.title("Scree Plot")
    pyplot.show()

    pca_components = input("Based on the Skree plot, enter the desired number of pca_components: ")

    pca_components = int(pca_components)
    print(pca_components)
    n_pca_components = PCA(n_components = pca_components)
    n_pca_components.fit(model_training_data)
    reduced_Model_training_data = n_pca_components.transform(model_training_data)
    reduced_Model_testing_data = n_pca_components.transform(model_test_data)
    reduced_Model_validation_data = n_pca_components.transform(model_validation_data)

    # get exact variability retained
    print("\nVar retained (%):", 
      np.sum(n_pca_components.explained_variance_ratio_ * 100))

    # verify shape after PCA
    print("Train images shape:", reduced_Model_training_data.shape)
    print("Test images shape: ", reduced_Model_testing_data.shape)
    print("Validation images shape: ", reduced_Model_validation_data.shape)


    return reduced_Model_training_data, reduced_Model_testing_data, reduced_Model_validation_data



  
def create_adversarial_pattern(input_image, input_label,img_pred):
  loss_object = tf.keras.losses.CategoricalCrossentropy()
  # tensor_image = input_image
  # input_image = tf.Variable(input_image)
  with tf.GradientTape() as tape:
    tape.watch(input_image)
    #prediction = pretrained_model(input_image)
    prediction = tf.convert_to_tensor(img_pred, dtype=tf.float32)
    input_label = tf.convert_to_tensor(input_label, dtype=tf.float32)
    
    print(prediction)
    print(input_label)
    loss = loss_object(input_label, prediction)
    print(loss)
     

  # Get the gradients of the loss w.r.t to the input image.
  gradient = tape.gradient(loss, input_image)
  print(gradient)
  # Get the sign of the gradients to create the perturbation
  signed_grad = tf.sign(gradient)
  return signed_grad
  
def display_images(image, description):
  _, label, confidence = get_imagenet_label(pretrained_model.predict(image))
  plt.figure()
  plt.imshow(image[0]*0.5+0.5)
  plt.title('{} \n {} : {:.2f}% Confidence'.format(description,
                                                   label, confidence*100))
  plt.show()


def main():
    
    # Define the One-hot Encoder
    ohe = preprocessing.OneHotEncoder()

    #Select the data set to model
    dataset_to_model = select_dataset_to_model(tfds.list_builders())


    #This section normalizes the data and trains the model
    tf.random.set_seed(10)

    (Model_training_data, Model_training_data_labels), (Model_test_data, Model_test_data_labels) = eval(tfds.list_builders()[dataset_to_model]+'.load_data()')
    
    pyplot.imshow(Model_training_data[0], cmap=pyplot.get_cmap('gray'))
    pyplot.show()
    
    Model_training_data = tf.keras.utils.normalize(Model_training_data,axis=1)
    Model_test_data = tf.keras.utils.normalize(Model_test_data,axis=1)

    Model_training_data = np.reshape(Model_training_data, (-1, 784))
    Model_test_data = np.reshape(Model_test_data, (-1, 784))
    
    (Model_training_data, valData, Model_training_data_labels, valLabels) = train_test_split(Model_training_data,Model_training_data_labels,
	test_size=0.5, random_state=4)
    
    # Add in PCA functionality
   # [Model_training_data, Model_test_data, valData] = determine_pca_components(Model_training_data,Model_test_data, valData)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())

    Model_classes = np.unique(Model_training_data_labels)
    print('model classes: ', Model_classes)
    total_number_of_classes = len(Model_classes)
    
   
    # show the sizes of each data split   
    print("training data points: {}".format(len(Model_training_data_labels)))
    print("validation data points: {}".format(len(valLabels)))
    print("testing data points: {}".format(len(Model_test_data_labels)))
    
    #Build the model
    nn_layers = int(input("Enter the number of layers desired: "))

    for index in range(nn_layers-1):
        nodes = input("Enter the desired number of nodes for layer_" + str(index+1) +": ")
        layer_name = "hidden_"+str(index+1)
        #model.add(tf.keras.layers.Dense(nodes, activation=tf.nn.relu,name=layer_name))
        model.add(tf.keras.layers.Dense(nodes, activation=tf.nn.leaky_relu,name=layer_name))
        print(layer_name)

    index +=2
    layer_name = "hidden_"+str(index)    
    nodes = input("Enter the desired number of nodes for layer_" + str(index) +": ")
    model.add(tf.keras.layers.Dense(nodes, activation=tf.nn.softmax,name=layer_name))
    print(layer_name)

    model.compile(optimizer='sgd',loss='sparse_categorical_crossentropy',metrics='accuracy')

    number_of_epochs = 15 
    model = fit_model(number_of_epochs, model,Model_training_data,Model_training_data_labels) 
    nn_predictions = model.predict(Model_test_data).argmax(axis=1)

    print("EVALUATION ON NN Model TESTING DATA")
    print(classification_report(Model_test_data_labels, nn_predictions))


    nn_cm = pd.DataFrame(confusion_matrix(Model_test_data_labels, nn_predictions), 
                      columns=Model_classes, index = Model_classes)

    # Seaborn's heatmap to better visualize the confusion matrix
    sns.heatmap(nn_cm, annot=True, fmt='d', linewidths = 0.30)
    pyplot.show()
    
    
    index_counter=[]
    labels_counter=[]

    [index_counter ,labels_counter] = class_indices(index_counter,labels_counter, Model_classes, Model_test_data_labels)
       
    #Capture the activations at each layer
    dataframe_dict = OrderedDict()
    [dataframe_dict, layer_nodes] = capture_activations(nn_layers, model, Model_test_data,dataframe_dict, index_counter ,labels_counter)

   #---------------------------------------------------------------------------------------------------
    test_labels = Model_test_data_labels
    test_labels = test_labels.reshape(-1,1)
    
    #Fit and transform test labels
    ohe.fit(test_labels)
    transformed_test_labels  = ohe.transform(test_labels).toarray()
    print(f'transformed with encoding: {transformed_test_labels[0:1]}')

    image = Model_test_data[0:1]
    reshaped_image = np.reshape(image, (28, 28))
    
    pyplot.imshow(reshaped_image, cmap=pyplot.get_cmap('gray'))
    pyplot.show()
    
    image_prediction = model.predict(image).argmax(axis=1)
    print("image prediction")
    print(image_prediction)
    
    #The last layer of the activation outputs has the softmax values
    image_probs_orig = dataframe_dict[6][0:1].to_numpy()
    image_pred = image_probs_orig
    image_probs = max(image_probs_orig)
    image_probs = image_probs[np.argmax(image_probs)]
    print(image_probs_orig)
      
    pyplot.figure()
    pyplot.imshow(reshaped_image * 0.5 + 0.5)  # To change [-1, 1] to [0,1]
    image_class = image_prediction
    class_confidence = image_probs
    
    pyplot.title('{} : {:.2f}% Confidence'.format(image_class, class_confidence*100))
    pyplot.show()

    # Get the input label of the image.    
    label = transformed_test_labels[0:1]
    print(label)

    tensor_image = tf.convert_to_tensor(image, dtype=tf.float32)
    perturbations = create_adversarial_pattern(tensor_image, label,image_pred)
    # pyplot.imshow(perturbations[0] * 0.5 + 0.5);  # To change [-1, 1] to [0,1]

    # epsilons = [0, 0.01, 0.1, 0.15]
    # descriptions = [('Epsilon = {:0.3f}'.format(eps) if eps else 'Input')
                    # for eps in epsilons]

    # for i, eps in enumerate(epsilons):
      # adv_x = image + eps*perturbations
      # adv_x = tf.clip_by_value(adv_x, -1, 1)
      # display_images(adv_x, descriptions[i])



        
if  __name__ == "__main__":
    main()
