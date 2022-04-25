import csv
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
from matplotlib import pyplot

#Load mnist data into variables
(train_X, train_Y), (test_X, test_Y) = mnist.load_data()

#Display basic information about the data and data structure
print('Training data shape: ', train_X.shape, train_Y.shape)

print('Testing data shape: ', test_X.shape, test_Y.shape)

classes = np.unique(test_Y)

nClasses = len(classes)

print('Total number of outputs: ', nClasses)

print('Output classes: ', classes)


# Resize the matrix from 28,28 to 784,1

flattened_data=test_X.reshape(len(test_X),784,1)

# Normalize the flattened data
n_data=flattened_data/255


#Step 1
index_counter=[]
#In the mnsist test set find the indices where each distinct number exists
for a in range(10):
    column=[]
    for b in range(len(test_Y)):
        if test_Y[b]==a:
            column.append(b)
    index_counter.append(column)
 
# Confirm results 
# print(index_counter[3])

# print(test_Y[18],test_Y[30],test_Y[32])

# print(index_counter[7])
# print(test_Y[0],test_Y[17],test_Y[34])

#Step 2
#For each number class group the "good" images from the "bad" ones

#Create csv files for each number class

for b in range(8,9):
    var1 = 'good_activations_'
    var2 = str(b)
    var3 = '.csv'
    var4= var1+var2+var3
    var5 = 'bad_activations_'
    var6 = var5+var2+var3
   
    good_indices=[]
    bad_indices=[]
    #for c in range(400,500):
    for c in range(500,600):
        pyplot.imshow(test_X[index_counter[b][c]], cmap=pyplot.get_cmap('gray'))
        # show the figure
        pyplot.show()
        # ask the user to enter g or b
        user_input = input("For good images enter g, for bad images enter b: ")
        if user_input=='g':
            good_indices.append(str(index_counter[b][c]))
        
       
        else:    
            bad_indices.append(str(index_counter[b][c]))
            
            
         
    with open(var4,'a+',newline='') as f:
        writer = csv.writer(f)    
        writer.writerow(good_indices)
    with open(var6,'a+',newline='') as f:
        writer = csv.writer(f)    
        writer.writerow(bad_indices)
    

 