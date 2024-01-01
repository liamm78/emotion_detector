import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from smile_detector.newLen import newLen
import imutils
from imutils import paths
import argparse
import cv2
import os
import numpy as np


dataset_dir = "C:/Users/liamm/Documents/Smile_Detector/Data"  # Replace with the actual path to your dataset directory
print("Hello")
data=[]
labels=[]
pos=[]
neg=[]




for imagePath in sorted(list(paths.list_images(dataset_dir))):
    image = cv2.imread(imagePath) # Gets a matrix of the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #Converts to gray scale
    image = imutils.resize(image, width=28) #Rescales to 28x28
    image = img_to_array(image) #Converts image to array
    data.append(image)
    
#Find the labels for these images
       # print(imagePath)
    label = imagePath.split(os.path.sep)[-2] #2nd to last
    labels.append(label)
    
# Scale the pixel intensities to 0-1 FROM 0-255
data= np.array(data, dtype=float) / 255.0
labels = np.array(labels)

#Convert labels into numbers
le = LabelEncoder().fit(labels)
labels = to_categorical(le.transform(labels), 2)

#Skew
classTotals = labels.sum(axis=0)
classWeight=dict() #ratio of how much images are positive/negative

for i in range(len(classTotals)):
    classWeight[i] = classTotals.max() / classTotals[i]

# MAKING THE MODEL #

trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.20, random_state=42) #X and Y
builder = newLen()

model = builder.build(width=28, height=28, depth=1, classes=2)

print("Shape of trainX:", trainX.shape)
print("Shape of trainY:", trainY.shape)
print("Shape of testX:", testX.shape)
print("Shape of testY:", testY.shape)

model.compile(loss=['binary_crossentropy'],optimizer='adam',metrics=['accuracy'])
H = model.fit(trainX, trainY, validation_data=(testX, testY), class_weight = classWeight, batch_size=64,epochs=10)
predict = model.predict(testX, batch_size=64)

print(H.history)
print(H.history['accuracy'])
#print(H.history['val_accuracy'])


model.save('model.h5')
#print(len(neg))
#print(len(pos))
        
    