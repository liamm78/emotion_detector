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



class VGG:
    def build(self, width, height, depth, classes): # 48 x 48 2 classes
        model=Sequential() # Sequential uses layers to build
        
        
        inputShape=(width, height, depth)
        
        # 1st CONV block
        model.add(Conv2D(32,(3,3),padding='same',input_shape=inputShape, activation='relu')) # 28x28
        model.add(BatchNormalization())
        model.add(AveragePooling2D(pool_size=(2, 2), strides=(2,2))) # 14x14
        
        # 2nd CONV block
        
        model.add(Conv2D(64,(3,3),padding='same',input_shape=inputShape, activation='relu')) # 28x28
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2))) # 14x14
        
        # 3rd CONV Block
        model.add(Conv2D(128,(3,3),padding='same',input_shape=inputShape, activation='relu')) # 28x28        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2))) # 14x14

        # first (and only) set of FC => ReLU layere
        model.add(Flatten()) #Flatten image to 1d vector
        model.add(Dense(256,kernel_regularizer=l2(0.01),bias_regularizer=l2(0.01))) #adds fully connected layer with 4096 
        model.add(Dropout(0.5))
        model.add(Dense(128,kernel_regularizer=l2(0.01),bias_regularizer=l2(0.01)))
        model.add(Dropout(0.5))

       
        model.add(Dense(classes)) #connects 500 neurons to 2 classes
        model.add(Activation('softmax')) #converts raw output into class probabiltiies. 
        
        return model