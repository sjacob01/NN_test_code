from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MSE
import tensorflow as tf

def build_model(width, height, depth, classes):
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

def generate_image_adversary(model, image, label, eps=2 / 255.0):
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

# load MNIST dataset and scale the pixel values to the range [0, 1]
print("[INFO] loading MNIST dataset...")
(trainX, trainY), (testX, testY) = mnist.load_data()
trainX = trainX / 255.0
testX = testX / 255.0
# add a channel dimension to the images
trainX = np.expand_dims(trainX, axis=-1)
testX = np.expand_dims(testX, axis=-1)
# one-hot encode our labels
trainY = to_categorical(trainY, 10)
testY = to_categorical(testY, 10)

# initialize our optimizer and model
print("[INFO] compiling model...")
opt = Adam(lr=1e-3)
model = build_model(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt,	metrics=["accuracy"])
# train the simple CNN on MNIST
print("[INFO] training network...")
model.fit(trainX, trainY,validation_data=(testX, testY),batch_size=64,epochs=10,verbose=1)
# make predictions on the testing set for the model trained on
# non-adversarial images
(loss, acc) = model.evaluate(x=testX, y=testY, verbose=0)
print("[INFO] loss: {:.4f}, acc: {:.4f}".format(loss, acc))

# loop over a sample of our testing images
for i in np.random.choice(np.arange(0, len(testX)), size=(10,)):
    print(i)
    # get the current image and label
    image = testX[i]
    label = testY[i]
    #print(label)
    # generate and adversary image for the current image and make a prediction on the adversary
    adversary = generate_image_adversary(model, image.reshape(1,28,28,1),label, eps=0.1)
    pred = model.predict(adversary)
    #print(adversary)
    #Scale both the original image and adversary to the range
    # [0,255] and convert them to an unsigned 8-bit integer
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
    
    # if the image prediction does not match the adversarial prediction then update the color
    if imagePred != adversaryPred:
        color = (0,0,255)
    # draw the predictions on the respective output images
    cv2.putText(image, str(imagePred), (2, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 0), 2)
    cv2.putText(adversary, str(adversaryPred), (2, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 2)
    
    # stack the two images horizontally and then show the original image and adversarial image
    output = np.hstack([image, adversary])
	
    cv2.imshow("FGSM Adversarial Images", output)
    cv2.waitKey(0)
#https://pyimagesearch.com/2021/03/01/adversarial-attacks-with-fgsm-fast-gradient-sign-method/
	

