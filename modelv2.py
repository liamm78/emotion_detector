import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from VGG import VGG
import imutils
from imutils import paths
import argparse
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

        #  C:/Users/liamm/Documents/Smile_Detector/Data"
dataset = "C:/Users/liamm/Documents/Smile_Detector/Data5E"
trainX = [] #features for train
testX = []
trainY = [] #labels for train
testY = []

for imagePath in sorted(list(paths.list_images(dataset))):
   image = cv2.imread(imagePath) # matrix of image
   image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   imutils.resize(image,28)
   image = img_to_array(image)
   
   
   
   split = imagePath.split(os.path.sep)[-3]
   label = imagePath.split(os.path.sep)[-2]
   if(split=="train"):   #divide the labels into train and test
       trainY.append(label)
       trainX.append(image)
   else:
       testY.append(label)
       testX.append(image)
   
trainX= np.array(trainX, dtype=float) / 255.0
testX = np.array(testX, dtype=float) /255.0


le = LabelEncoder().fit(trainY)
trainY = to_categorical(le.transform(trainY), num_classes=5)
testY = to_categorical(le.transform(testY), num_classes=5)

from sklearn.utils import class_weight

classe = np.arange(5)

y = np.argmax(trainY, axis=1)

weight = class_weight.compute_class_weight(class_weight='balanced', classes=classe, y=y)
# Convert the weight array to a dictionary using the dict and zip functions
weight_dict = dict(zip(classe, weight))
print(weight_dict)
    
builder = VGG()
model = builder.build(width=48, height=48, depth=1, classes=5)

print("Shape of trainX:", trainX.shape)
print("Shape of trainY:", trainY.shape)
print("Shape of testX:", testX.shape)
print("Shape of testY:", testY.shape)

model.compile(loss=['categorical_crossentropy'],optimizer='adam',metrics=['accuracy'])
H = model.fit(trainX, trainY, validation_data=(testX, testY), class_weight = weight_dict, batch_size=64,epochs=30)
predict = model.predict(testX, batch_size=64)

print(H.history)
print(H.history['accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()
#print(H.history['val_accuracy'])


model.save('modelv3.h5')



   #data.append(image)

#print(len(trainX))
#print(len(test))
