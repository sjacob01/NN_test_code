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




def main():
    
  #Select the data set to model
  dataset_to_model = select_dataset_to_model(tfds.list_builders())



if __name__ == "__main__":
    main()


