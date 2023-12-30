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
data = []
labels = []

for imagePath in sorted(list(paths.list_images(dataset))):
   image = cv2.imread(imagePath) # matrix of image
   image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   imutils.resize(image,28)
   image = img_to_array(image)
   data.append(image)
   
   label = imagePath.split(os.path.sep)[-2] #2nd to last
   labels.append(label)
   
data= np.array(data, dtype=float) / 255.0
labels = np.array(labels)

#Convert labels into numbers
le = LabelEncoder().fit(labels)
labels = to_categorical(le.transform(labels), 5)

#Skew
classTotals = labels.sum(axis=0)
classWeight=dict() #ratio of how much images are positive/negative

for i in range(len(classTotals)):
    classWeight[i] = classTotals.max() / classTotals[i]

# MAKING THE MODEL #

trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.20, random_state=42) #X and Y
    
builder = VGG()
model = builder.build(width=48, height=48, depth=1, classes=5)

print("Shape of trainX:", trainX.shape)
print("Shape of trainY:", trainY.shape)
print("Shape of testX:", testX.shape)
print("Shape of testY:", testY.shape)

model.compile(loss=['categorical_crossentropy'],optimizer='adam',metrics=['accuracy'])
H = model.fit(trainX, trainY, validation_data=(testX, testY), class_weight = classWeight, batch_size=64,epochs=30)
predict = model.predict(testX, batch_size=64)

print(H.history)
print(H.history['accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()
#print(H.history['val_accuracy'])
#print(H.history['val_accuracy'])


model.save('model5E.h5')

