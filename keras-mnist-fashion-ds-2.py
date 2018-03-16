##!/usr/bin/env python3
"""
Created on Mon Mar 12 21:36:39 2018
" TSB visual recognition of keras fashion dataset with SGD
" fashion dataset has images of various types of clothing and accessories
"show some of the images and predictions
"1st iteration created the model (commented out, explained in comments),
"this iteration calls and uses it
@author: hsantanam
"""
#import libraries and modules
import numpy as np
np.random.seed(123)  #for reproducability

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt

#load the mnist fashion dataset into training and testing sets and print their dimensions
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

#preprocess the data
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# preprocess class lables
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

#define the model architecture
model = Sequential()
model.add(Dense(units=128, activation="relu", input_shape=(784,)))
model.add(Dense(units=128, activation="relu"))
model.add(Dense(units=128, activation="relu"))
model.add(Dense(units=10, activation="softmax"))

#compile the model
#sgd = SGD(lr=0.001)
model.compile(loss="categorical_crossentropy", optimizer=SGD(0.001), metrics=["accuracy"])

#fit the model to the data
model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=1)

#sbelow, ave the model so it can be applied to other datasets if needed - 1st time
#model.save("mnist_fashion_ds-1.h5")

#note: before using line below, line above (model.save) must be done
#...since you can't load something that doesn't exist :)
# and once you do, you can either re-save the model each time or simply load it like below
model.load_weights("mnist_fashion_ds-1.h5")

#evaluate the model on test data
#accuracy = model.evaluate(x = x_test, y=y_test, batch_size=32)
#print("Accuracy: ", accuracy[1])
#
#key-value pair, to show what each prediction result means to a human
#for easy readibility...e.g.,'Dress' easier to read than '3'
d = {0:'T-shirt/top',
     1: 'Trouser',
     2: 'Pullover',
     3: 'Dress',
     4: 'Coat',
     5: 'Sandal',
     6: 'Shirt',
     7: 'Sneaker',
     8: 'Bag',
     9: 'Ankle boot'}

#print a range of images from the test image data and show code's image predition
for x in range(100,4000,200):                   #from 100 to 4000, in increments of 200
    img = x_test[x]                             #set img = an image from the array as selected above 
    test_img = img.reshape((1,784))
    img_class = model.predict_classes(test_img)  #predict the image from this image file instance
    classname = img_class[0]                     #set classname = prediction for that image  
    print("Class: ", classname, d[classname])
    
    img = img.reshape((28, 28))                  #reshape to 28x28 pixels 
    plt.imshow(img)                              #show the image
   
    plt.title(classname)                         #set & print the title to screen
    plt.show()                          