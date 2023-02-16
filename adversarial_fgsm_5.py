import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import umap
import umap.plot
import utils
import seaborn as sns
import tensorflow_datasets as tfds
import ssl
import array
ssl._create_default_https_context = ssl._create_unverified_context
from matplotlib import pyplot
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
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
from tensorflow.keras.losses import SparseCategoricalCrossentropy, MSE
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import cv2




## ----------------------------------------------------------------------------------------------

def build_cnn_model(width, height, depth, classes):
    # initialize the model along with the input shape
    model = Sequential()
    inputShape = (height, width, depth)
    chanDim = -1
    # first CONV => RELU => BN layer set
    model.add(Conv2D(32, (3, 3), strides=(2, 2), padding="same",
        input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    # second CONV => RELU => BN layer set
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    # first (and only) set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    # softmax classifier
    model.add(Dense(classes))
    model.add(Activation("softmax"))
    # return the constructed network architecture
    return model


def generate_image_adversary(model, image, label,eps):
	# cast the image
    
	image = tf.cast(image, tf.float32)
    # record our gradients
	with tf.GradientTape() as tape:
		# explicitly indicate that our image should be tacked for
		# gradient updates
		tape.watch(image)
		# use our model to make predictions on the input image and
		# then compute the loss
		pred = model(image)
		loss = MSE(label, pred)
    # calculate the gradients of loss with respect to the image, then
	# compute the sign of the gradient
	gradient = tape.gradient(loss, image)
	signedGrad = tf.sign(gradient)
	# construct the image adversary
	adversary = (image + (signedGrad * eps)).numpy()

	# return the image adversary to the calling function
	return adversary

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



def create_nn_model():
    #Build the model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
           
    nn_layers = int(input("Enter the number of layers desired: "))

    for index in range(nn_layers-1):
        nodes = input("Enter the desired number of nodes for layer_" + str(index+1) +": ")
        layer_name = "hidden_"+str(index+1)
        model.add(tf.keras.layers.Dense(nodes, activation=tf.nn.leaky_relu,name=layer_name))
        print(layer_name)

    index +=2
    layer_name = "hidden_"+str(index)    
    nodes = input("Enter the desired number of nodes for layer_" + str(index) +": ")
    model.add(tf.keras.layers.Dense(nodes, activation=tf.nn.softmax,name=layer_name))
    print(layer_name)

   
    return model, nn_layers




def main():
   
    #Select the data set to model
    dataset_to_model = select_dataset_to_model(tfds.list_builders())

    #This section normalizes the data and trains the model
    tf.random.set_seed(10)

    (Model_training_data, Model_training_data_labels), (Model_test_data, Model_test_data_labels) = eval(tfds.list_builders()[dataset_to_model]+'.load_data()')
    
    Model_classes = np.unique(Model_training_data_labels)
    print('model classes: ', Model_classes)
    total_number_of_classes = len(Model_classes)
    
 
    (Model_training_data, valData, Model_training_data_labels, valLabels) = train_test_split(Model_training_data,Model_training_data_labels,
	test_size=0.4, random_state=4)
 
    (Model_training_data, advData, Model_training_data_labels, advLabels) = train_test_split(Model_training_data,Model_training_data_labels,
	test_size=0.16, random_state=4)
    # show the sizes of each data split   
    print("training data points: {}".format(len(Model_training_data_labels)))
    print("validation data points: {}".format(len(valLabels)))
    print("adversarial data points: {}".format(len(advLabels)))
    print("testing data points: {}".format(len(Model_test_data_labels)))
 
    
    Model_training_data = Model_training_data/255.0
    advData=advData/255.0
    Model_test_data = Model_test_data/255.0
    
    Model_test_data_2 = Model_test_data
    
    # add a channel dimension to the images
    Model_training_data = np.expand_dims(Model_training_data, axis=-1)
    advData = np.expand_dims(advData, axis=-1)
    Model_test_data = np.expand_dims(Model_test_data, axis=-1)
    

    # one-hot encode our labels
    ohe_Model_training_data_labels = to_categorical(Model_training_data_labels, 10)
    ohe_Model_adv_data_labels = to_categorical(advLabels, 10)
    ohe_Model_test_data_labels = to_categorical(Model_test_data_labels, 10)

 #---------Build CNN model ----------------------------------------------
    # initialize our optimizer and model
    print("[INFO] compiling CNN model...")
    opt = Adam(lr=1e-3)
    cnn_model = build_cnn_model(width=28, height=28, depth=1, classes=10)
    cnn_model.compile(loss="categorical_crossentropy", optimizer=opt,	metrics=["accuracy"])
    # train the simple CNN on MNIST
    print("[INFO] training network...")
    cnn_model.fit(Model_training_data, ohe_Model_training_data_labels,validation_data=(Model_test_data, ohe_Model_test_data_labels),batch_size=64,epochs=10,verbose=1)
    # make predictions on the testing set for the model trained on
    # non-adversarial images
    (loss, acc) = cnn_model.evaluate(x=Model_test_data, y=ohe_Model_test_data_labels, verbose=0)
    print("[INFO] loss: {:.4f}, acc: {:.4f}".format(loss, acc))


#---------Build Basic NN model---------------------------------------------------------------------------------
   
    [model, nn_layers]= create_nn_model()
    model.compile(optimizer='sgd',loss='sparse_categorical_crossentropy',metrics='accuracy')
    
    number_of_epochs = 15 
    Model_training_data = np.reshape(Model_training_data, (-1, 784))
    Model_test_data = np.reshape(Model_test_data, (-1, 784))
   
    # Fit the model
    model = fit_model(number_of_epochs, model,Model_training_data, Model_training_data_labels) 
    (loss, acc) = model.evaluate(x=Model_test_data, y=Model_test_data_labels)
    print(loss)
    print(acc)
  
    nn_predictions = model.predict(Model_test_data).argmax(axis=1)

    print("EVALUATION ON NN Model TESTING DATA")
    print(classification_report(Model_test_data_labels, nn_predictions))


    nn_cm = pd.DataFrame(confusion_matrix(Model_test_data_labels, nn_predictions), 
                      columns=Model_classes, index = Model_classes)

    # Seaborn's heatmap to better visualize the confusion matrix
    sns.heatmap(nn_cm, annot=True, fmt='d', linewidths = 0.30)
    pyplot.show()

#---------------------------------------------------------------------------------
  
#---------------------------------------------------------------------------------
# Create an adversarial image using the CNN model
    adv_arr = [] 
   # get the current image and label
    for i in range(len(advData)):
        image = advData[i]
        image_2 = image
        label = ohe_Model_adv_data_labels[i]
     
    # generate and adversary image for the current image and make a prediction on the adversary
        adversary = generate_image_adversary(cnn_model, image.reshape(1,28,28,1),label, eps=0.35)
        pred = cnn_model.predict(adversary)
    
    #Scale both the original image and adversary to the range
    # [0,255] and convert them to an unsigned 8-bit integer
        adversary_2 =adversary.reshape(28, 28,1)
        adversary_3 = adversary.reshape(28,28)
       
        adv_arr.append(adversary_3)
        adversary = adversary.reshape((28,28))*255
        adversary = np.clip(adversary, 0, 255).astype("uint8")
        image = image.reshape((28, 28)) * 255
        image = image.astype("uint8")
        # convert the image and adversarial image from grayscale to three
        # channel (so we can draw on them)
        image = np.dstack([image] * 3)
        adversary = np.dstack([adversary] * 3)
        # resize the images so we can better visualize them
        image = cv2.resize(image, (96, 96))
        adversary = cv2.resize(adversary, (96, 96))
     
    # determine the predicted label for both the original image and adversarial image
        imagePred = label.argmax()
        adversaryPred = pred[0].argmax()
        color = (0, 255, 0)

    # # if the image prediction does not match the adversarial prediction then update the color
        # if imagePred != adversaryPred:
            # color = (0,0,255)
        # # draw the predictions on the respective output images
        # cv2.putText(image, str(imagePred), (2, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 0), 2)
        # cv2.putText(adversary, str(adversaryPred), (2, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 2)
        
        # # stack the two images horizontally and then show the original image and adversarial image
        # output = np.hstack([image, adversary])
    
        # #print('Correct and adversarial images')
        # cv2.imshow("CNN FGSM Adversarial Images", output)
        # cv2.waitKey(0)
# # --------------------------------------------------------------------------

        # Show the prediction that the basic neural network makes for the adversarial image 
        adversary_2 = np.reshape(adversary_2, (-1, 784))        
        adv_pred = model.predict(adversary_2)
       
        # adversary_2  = adversary_2 .reshape((28,28))*255
        # adversary_2  = np.clip(adversary_2 , 0, 255).astype("uint8")
        # image_2 = image_2.reshape((28, 28)) * 255
        # image_2= image_2.astype("uint8")
        # # convert the image and adversarial image from grayscale to three
        # # channel (so we can draw on them)
        # image_2 = np.dstack([image_2] * 3)
        # adversary_2  = np.dstack([adversary_2 ] * 3)
        # # resize the images so we can better visualize them
        # image_2 = cv2.resize(image_2, (96, 96))
        # adversary_2  = cv2.resize(adversary_2 , (96, 96))
    
        # # determine the predicted label for both the original image and adversarial image
        # imagePred = label.argmax()
        # adversaryPred = adv_pred[0].argmax()
        # color = (0, 255, 0)

        # # if the image prediction does not match the adversarial prediction then update the color
        # if imagePred != adversaryPred:
            # color = (0,0,255)
        # # draw the predictions on the respective output images
        # cv2.putText(image_2, str(imagePred), (2, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 0), 2)
        # cv2.putText(adversary_2, str(adversaryPred), (2, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 2)
        
        # # stack the two images horizontally and then show the original image and adversarial image
        # output = np.hstack([image_2, adversary_2])
        
        # print('NN Correct and adversarial images')
        # print('The original value is :',advLabels[i],' The predicted value is :', adv_pred.argmax()) 
        
        # cv2.imshow("NN FGSM Adversarial Images", output)
        # cv2.waitKey(0)
    
  
  
  # --------------------------------------------------------------------------------------------------
    index_counter=[]
    labels_counter=[]
    
    [index_counter ,labels_counter] = class_indices(index_counter,labels_counter, Model_classes, advLabels)
    adv_arr = np.asarray(adv_arr)
    
    print(adv_arr.shape)
   
    test_adv_data= np.append(Model_test_data_2 , adv_arr, axis =0)
    
    dataframe_dict = OrderedDict()
    data_adv = np.reshape(adv_arr, (-1, 784))
    # [dataframe_dict, layer_nodes] = capture_activations(nn_layers, model, data_adv,dataframe_dict, index_counter ,labels_counter)

    # for c in range(1,nn_layers):


        # v= index_counter[0]+index_counter[1]+index_counter[2]+index_counter[3]+index_counter[4]+index_counter[5]+index_counter[6]+index_counter[7]+index_counter[8]+index_counter[9]
        # v_label=labels_counter[0]+labels_counter[1]+labels_counter[2]+labels_counter[3]+labels_counter[4]+labels_counter[5]+labels_counter[6]+labels_counter[7]+labels_counter[8]+labels_counter[9]
        # arr=np.array(v_label)

        # mapper = umap.UMAP().fit(dataframe_dict[c].iloc[v])        
        # p=umap.plot.points(mapper, labels=arr,color_key_cmap='Paired', background='black')
        # var7 = 'ADVLayer_'+str(c)+'_nodes_'+str(layer_nodes[c-1])
        # umap.plot.plt.title(var7)
        # umap.plot.plt.show()
  
  
  #-------------------------------------------------------------------------------------------------
    (loss, acc) = model.evaluate(x=data_adv, y=advLabels)
    print(loss)
    print(acc)
  
    nn_predictions = model.predict(data_adv).argmax(axis=1)

    print("EVALUATION ON NN Model adversarial DATA")
    print(classification_report(advLabels, nn_predictions))


    nn_cm = pd.DataFrame(confusion_matrix(advLabels, nn_predictions), 
                      columns=Model_classes, index = Model_classes)

    # Seaborn's heatmap to better visualize the confusion matrix
    sns.heatmap(nn_cm, annot=True, fmt='d', linewidths = 0.30)
    pyplot.show()
    
# -----------------------------------------------------------------------------------------

# Combine the adversarial data with the validation data
    
    test_adv_labels = np.append(Model_test_data_labels, advLabels)
    test_adv_data = np.reshape(test_adv_data, (-1, 784))

    (loss, acc) = model.evaluate(x=test_adv_data, y=test_adv_labels)
    print(loss)
    print(acc)
    
    nn_predictions = model.predict(test_adv_data).argmax(axis=1)

    print("EVALUATION ON NN Model test and adversarial DATA")
    print(classification_report(test_adv_labels, nn_predictions))


    nn_cm = pd.DataFrame(confusion_matrix(test_adv_labels, nn_predictions), 
                      columns=Model_classes, index = Model_classes)

    # Seaborn's heatmap to better visualize the confusion matrix
    sns.heatmap(nn_cm, annot=True, fmt='d', linewidths = 0.30)
    pyplot.show()

#--------------------------------------------------------------------

if  __name__ == "__main__":
    main()
