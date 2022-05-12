import pandas as pd
import tensorflow as tf
import numpy as np
import math
import utils
import seaborn as sns
from sklearn.manifold import TSNE
from matplotlib import pyplot
from numpy import exp
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score
from keract import get_activations, display_activations, display_heatmaps



tf.random.set_seed(10)
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()


X_train = tf.keras.utils.normalize(X_train,axis=1)
X_test = tf.keras.utils.normalize(X_test,axis=1)

X_test=X_test.reshape(len(X_test),784)


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(588, activation=tf.nn.relu,name="hidden_1"))
model.add(tf.keras.layers.Dense(392, activation=tf.nn.relu,name="hidden_2"))
model.add(tf.keras.layers.Dense(196, activation=tf.nn.relu,name="hidden_3"))
model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu,name="hidden_4"))
model.add(tf.keras.layers.Dense(32, activation=tf.nn.relu,name="hidden_5"))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax,name="hidden_6"))

model.compile(optimizer='sgd',loss='sparse_categorical_crossentropy',metrics='accuracy')
#model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics='accuracy')

#model.compile(loss='binary_crossentropy', optimizer='sgd', metrics='accuracy')


model.fit(X_train,Y_train,epochs=10)


activations = get_activations(model, X_train[1:2], auto_compile=True)
[print(k, '->', v.shape, '- Numpy array') for (k, v) in activations.items()]

classes = np.unique(Y_train)
print(classes)


# #-----------------------------------------------------------------------------------
# # Show two example images
# fig, ax = pyplot.subplots(1,2)

# ax[0].imshow(X_test[11,:].reshape(28,28), 'Greys')
# ax[1].imshow(X_test[15,:].reshape(28,28), 'Greys')
# ax[0].set_title(str(Y_test[11]))
# ax[1].set_title(str(Y_test[15]))

# pyplot.show()

#------------------------------------------------------------------------------------
# Two dimensions for each of our images
n_components = 2
tsne = TSNE(n_components)
tsne_result = tsne.fit_transform(X_test)

print(len(tsne_result))

# Plot the result of our TSNE with the label color coded
tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label':Y_test})


palette = sns.color_palette("bright", 10)
fig, ax = pyplot.subplots(1)
sns.scatterplot(x='tsne_1', y='tsne_2', data=tsne_result_df, legend='full', palette =palette, hue='label', ax=ax, s=120)
lim = (tsne_result.min()-5, tsne_result.max()+5)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_aspect('equal')
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

pyplot.show()