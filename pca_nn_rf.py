import pandas as pd
import tensorflow as tf
import numpy as np
import math
import umap
import umap.plot
import utils
import seaborn as sns
import tensorflow_datasets as tfds
import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from matplotlib import pyplot
from numpy import exp
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from keract import get_activations, display_activations, display_heatmaps
from tensorflow.keras.datasets import *
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import classification_report
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
#importing Seaborn's to use the heatmap 
import seaborn as sns
from mlxtend.plotting import plot_decision_regions
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler


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



def fit_model(number_of_epochs, model,training_data,training_data_labels):
#Train the model on the desired data set
   
    model.fit(training_data,training_data_labels,epochs=number_of_epochs)
    score = model.evaluate(training_data,training_data_labels)
        
    if score[1] < .98:
        number_of_epochs += 10
        return fit_model(number_of_epochs, model,training_data,training_data_labels)
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
     


def capture_activations(nn_layers, model, test_data, dataframe_dict,index_counter ,labels_counter):

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
         
        for e in range(len(test_data)):
            
            image=test_data[e:e+1]
            activations = get_activations(model,image)
            layer_activations = activations[h].T
            layer_activations = np.reshape(layer_activations,nodes)
            data.append(layer_activations)
            
       
        activation_dataframe = pd.DataFrame(data)
        
        activation_dataframe.columns = [activation_col_names]          
      
        activation_dataframe.to_csv(L3, index=False)
        
        dataframe_dict[c]= activation_dataframe
        layer_nodes.append(nodes)
        
        
                
    return dataframe_dict, layer_nodes
       

def determine_pca_components(training_data, test_data, validation_data):
    #This function reduces the dimensionality of the data
    
    
    pca_784 = PCA(n_components=784)
    pca_784.fit(training_data)

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
    n_pca_components.fit(training_data)
    reduced_Model_training_data = n_pca_components.transform(training_data)
    reduced_Model_testing_data = n_pca_components.transform(test_data)
    reduced_Model_validation_data = n_pca_components.transform(validation_data)
    
    # get exact variability retained
    print("\nVar retained (%):", 
      np.sum(n_pca_components.explained_variance_ratio_ * 100))
      
    # verify shape after PCA
    print("Train images shape:", reduced_Model_training_data.shape)
    print("Test images shape: ", reduced_Model_testing_data.shape)

    
    return reduced_Model_training_data, reduced_Model_testing_data, reduced_Model_validation_data


def main():
    
    #Select the data set to model
    dataset_to_model = select_dataset_to_model(tfds.list_builders())


    #This section normalizes the data and trains the model
    tf.random.set_seed(10)

    (Model_training_data, Model_training_data_labels), (Model_test_data, Model_test_data_labels) = eval(tfds.list_builders()[dataset_to_model]+'.load_data()')
    
    # Split the modeling training set for future use
    (trainData, valData, trainLabels, valLabels) = train_test_split(Model_training_data,Model_training_data_labels,
	test_size=0.5, random_state=4)
    
    trainData = tf.keras.utils.normalize(trainData,axis=1)
    Model_test_data = tf.keras.utils.normalize(Model_test_data,axis=1)
      
    trainData = np.reshape(trainData, (-1, 784))
    valData = np.reshape(valData, (-1, 784))
    Model_test_data = np.reshape(Model_test_data, (-1, 784))
    
    # Add in PCA functionality
    [trainData, Model_test_data, valData] = determine_pca_components(trainData,Model_test_data,valData)
   
    nn_model = tf.keras.models.Sequential()
    nn_model.add(tf.keras.layers.Flatten())
    
    Model_classes = np.unique(trainLabels)
    print('model classes: ', Model_classes)
    total_number_of_classes = len(Model_classes)
    
    
    nn_layers = int(input("Enter the number of layers desired: "))

    for index in range(nn_layers-1):
        nodes = input("Enter the desired number of nodes for layer_" + str(index+1) +": ")
        layer_name = "hidden_"+str(index+1)
        #model.add(tf.keras.layers.Dense(nodes, activation=tf.nn.relu,name=layer_name))
        nn_model.add(tf.keras.layers.Dense(nodes, activation=tf.nn.leaky_relu,name=layer_name))
        print(layer_name)
        
    index +=2
    layer_name = "hidden_"+str(index)    
    nodes = input("Enter the desired number of nodes for layer_" + str(index) +": ")
    nn_model.add(tf.keras.layers.Dense(nodes, activation=tf.nn.softmax,name=layer_name))
    print(layer_name)

    nn_model.compile(optimizer='sgd',loss='sparse_categorical_crossentropy',metrics='accuracy')

    number_of_epochs = 10 
    nn_model = fit_model(number_of_epochs, nn_model,trainData,trainLabels) 
    nn_predictions = nn_model.predict(Model_test_data).argmax(axis=1)
   
    
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
    
    dataframe_dict = OrderedDict()
    [dataframe_dict, layer_nodes] = capture_activations(nn_layers, nn_model, Model_test_data,dataframe_dict, index_counter ,labels_counter)
    
    print(dataframe_dict[1])
    for c in range(1,nn_layers):
            
                
        v= index_counter[0]+index_counter[1]+index_counter[2]+index_counter[3]+index_counter[4]+index_counter[5]+index_counter[6]+index_counter[7]+index_counter[8]+index_counter[9]
        v_label=labels_counter[0]+labels_counter[1]+labels_counter[2]+labels_counter[3]+labels_counter[4]+labels_counter[5]+labels_counter[6]+labels_counter[7]+labels_counter[8]+labels_counter[9]
        arr=np.array(v_label)
        
        mapper = umap.UMAP().fit(dataframe_dict[c].iloc[v])        
        p=umap.plot.points(mapper, labels=arr,color_key_cmap='Paired', background='black')
        var7 = 'Layer_'+str(c)+'_nodes_'+str(layer_nodes[c-1])
        umap.plot.plt.title(var7)
        umap.plot.plt.show()
                
    # #Initial RF section
    # # Split the modeling data set. From one half extract the activations
    # # Train those activations in RF
    # # Then identify the results on the known test set
    

    # show the sizes of each data split
    print("Original training data points: {}".format(len(Model_training_data)))
    print("training data points: {}".format(len(trainLabels)))
    print("validation data points: {}".format(len(valLabels)))
    print("testing data points: {}".format(len(Model_test_data_labels)))
    
    
    index_counter=[]
    labels_counter=[]
         
    [index_counter ,labels_counter] = class_indices(index_counter,labels_counter, Model_classes, valLabels)
    
    new_dataframe_dict = OrderedDict()
    [new_dataframe_dict, layer_nodes] = capture_activations(nn_layers, nn_model, valData,new_dataframe_dict, index_counter ,labels_counter)
    
    #process each row in the data frame
    clf=RandomForestClassifier(n_estimators=100)
    
    for index in range(1,nn_layers):
        clf.fit(new_dataframe_dict[index], valLabels)
    
        rf_pred=clf.predict(dataframe_dict[index])
    
        print("Accuracy:", metrics.accuracy_score(Model_test_data_labels, rf_pred))
    
        print("EVALUATION ON NN and RF TESTING DATA")
        print(classification_report(Model_test_data_labels, rf_pred))
          
    
        rf_cm = pd.DataFrame(confusion_matrix(Model_test_data_labels, rf_pred), 
                      columns=Model_classes, index = Model_classes)
                      
        # Seaborn's heatmap to better visualize the confusion matrix
        sns.heatmap(rf_cm, annot=True, fmt='d', linewidths = 0.30)
        pyplot.show()
    
 
    # # Implement KNN portion
        # kVals = range(1, 10, 2)
        # accuracies = []
    # # loop over various values of `k` for the k-Nearest Neighbor classifier
        # for k in range(1, 10, 2):
            # # train the k-Nearest Neighbor classifier with the current value of `k`
            # model = KNeighborsClassifier(n_neighbors=k)
            # model.fit(new_dataframe_dict[index], valLabels)

            # # evaluate the model and update the accuracies list
            # score = model.score(new_dataframe_dict[index], valLabels)
            # print("k=%d, accuracy=%.2f%%" % (k, score * 100))
            # accuracies.append(score)
            
        # # find the value of k that has the largest accuracy
        # i = int(np.argmax(accuracies))
        # print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[i],
        # accuracies[i] * 100))
        
        # # # re-train our classifier using the best k value and predict the labels of the
        # # # test data
        # #model = KNeighborsClassifier(n_neighbors=kVals[i])
        # model = KNeighborsClassifier(n_neighbors=kVals[i])
        # model.fit(new_dataframe_dict[index], valLabels)
        
        # knn_predictions = model.predict(dataframe_dict[index])
        
        # print("Accuracy:", metrics.accuracy_score(Model_test_data_labels, knn_predictions))
        
        # print("EVALUATION ON KNN TESTING DATA")
        # print(classification_report(Model_test_data_labels, knn_predictions))
        
        # rf_cm = pd.DataFrame(confusion_matrix(Model_test_data_labels, knn_predictions), 
                      # columns=Model_classes, index = Model_classes)
                      
        # # Seaborn's heatmap to better visualize the confusion matrix
        # sns.heatmap(rf_cm, annot=True, fmt='d', linewidths = 0.30)
        # pyplot.show()

 
if  __name__ == "__main__":
    main()
