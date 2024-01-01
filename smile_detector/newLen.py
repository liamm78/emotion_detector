import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dense
from keras import backend as K


class newLen:
    def build(self, width, height, depth, classes): # 28 x 28 2 classes
        model=Sequential() # Sequential uses layers to build
        
        
        inputShape=(width, height, depth)
        
        # 1st CONV layer
        model.add(Conv2D(20,(5,5),padding='same',input_shape=inputShape)) # 28x28
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2))) # 14x14
        
        # 2nd CONV layer
        model.add(Conv2D(50,(5,5),padding='same',input_shape=inputShape)) #14x14
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2))) # 7x7  (2, 2) halfs it
        
        # first (and only) set of FC => ReLU layere
        model.add(Flatten()) #Flatten image to 1d vector
        model.add(Dense(500)) #adds fully connected layer with 500 neurons. leanrs features
        model.add(Activation('relu'))
        
        # use softmax 
        
        model.add(Dense(classes)) #connects 500 neurons to 2 classes
        model.add(Activation('softmax')) #converts raw output into class probabiltiies. 
        
        return model