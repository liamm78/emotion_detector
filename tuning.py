import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
import imutils
from imutils import paths
import argparse
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras import backend as K
from keras.regularizers import l2
from keras.optimizers import Adam



def model_builder(hp): # 48 x 48 2 classes
        model=Sequential() # Sequential uses layers to build
        inputShape=(48, 48, 1)
        
        hp_activation = hp.Choice('activation', values=['relu','tanh','sigmoid'])
        hp_layer1 = hp.Int('layer1',min_value=1,max_value=1000,step=100)
        hp_layer2 = hp.Int('layer2',min_value=1,max_value=1000,step=100)
        hp_dropout = hp.Float('dropout',min_value=0,max_value=1,step=0.1)
        hp_learningRate=hp.Choice('learning_rate',values=[1e-2,1e-3,1e-4])
        #hp_optimizer = hp.Choice('optimizer', values=['adam', 'sgd', 'rmsprop'])

        # 1st CONV block
        model.add(Conv2D(32,(3,3),padding='same',input_shape=inputShape, activation=hp_activation)) # 28x28
        model.add(BatchNormalization())
        model.add(AveragePooling2D(pool_size=(2, 2), strides=(2,2))) # 14x14
        
        # 2nd CONV block
        
        model.add(Conv2D(64,(3,3),padding='same',input_shape=inputShape, activation=hp_activation)) # 28x28
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2))) # 14x14
        
        # 3rd CONV Block
        model.add(Conv2D(128,(3,3),padding='same',input_shape=inputShape, activation=hp_activation)) # 28x28        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2))) # 14x14

        # first (and only) set of FC => ReLU layere
        model.add(Flatten()) #Flatten image to 1d vector
        model.add(Dense(units=hp_layer1,kernel_regularizer=l2(0.01),bias_regularizer=l2(0.01),activation=hp_activation)) #adds fully connected layer with 4096 
        model.add(Dropout(hp_dropout))
        model.add(Dense(units=hp_layer2,kernel_regularizer=l2(0.01),bias_regularizer=l2(0.01),activation=hp_activation))
        model.add(Dropout(hp_dropout))

       
        model.add(Dense(5)) #connects 500 neurons to 2 classes
        model.add(Activation('softmax')) #converts raw output into class probabiltiies. 
        
        model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate = hp_learningRate),metrics=['accuracy'])
        return model
    
    
dataset = "C:/Users/liamm/Documents/Smile_Detector/Data5E"
data = []
labels = []
# Put labels and data in an array
for imagePath in sorted(list(paths.list_images(dataset))):
   image = cv2.imread(imagePath) # matrix of image
   image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   imutils.resize(image,28)
   image = img_to_array(image)
   data.append(image)
    
   # Categorize the image labels 
   label = imagePath.split(os.path.sep)[-2] #2nd to last
   labels.append(label)
   # Normalize the pixel values
data= np.array(data, dtype=float) / 255.0
labels = np.array(labels)

#Convert labels into numbers
le = LabelEncoder().fit(labels)
labels = to_categorical(le.transform(labels), 5)

#Account for Skew and Bias
classTotals = labels.sum(axis=0)
classWeight=dict() 

for i in range(len(classTotals)):
    classWeight[i] = classTotals.max() / classTotals[i]

# MAKING THE MODEL #

trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.20, random_state=42) #X and Y
    

#model_builder = build()


import keras_tuner as kt

tuner = kt.Hyperband(model_builder,objective='val_accuracy',max_epochs=10,directory='dir',factor=3)
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=3)
tuner.search(trainX,trainY,validation_data=(testX,testY),epochs=50,callbacks=[stop_early],class_weight=classWeight)


