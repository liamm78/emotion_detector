import cv2
import numpy as np
import imutils
from keras.preprocessing.image import img_to_array

from imutils import paths
from keras.models import load_model

labels = ['angry','disgust','fear','happy','neutral','sad','surprised']

def preprocess(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #Converts to gray scale
        image = cv2.resize(image, (28,28)) #Rescales to 28x28
        image = image.astype("float") / 255.0
        image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=-1)  # Add an extra dimension for the channel
        return image

camera = cv2.VideoCapture(0)

while True:
        grabbed, frame = camera.read()
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        model = load_model('model5E.h5')
        face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
        frame_clone = frame.copy()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
        print(faces)
        for (x, y, width, height) in faces:   # Get the region of interest
                cv2.rectangle(frame,(x,y),(x+width,y+height),(0,255,0),2)
             
                roi = gray[y:y+height,x:x+width] # GETS THE FACE (REGION OF INTEREST) 
                
                roi = cv2.resize(roi, (48,48))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0) #our actual input
                cv2.imshow('Face', frame_clone)
                
                prob = model.predict(roi)[0]
                print(prob)
                result = np.argmax(prob) #find max index of array
                print(labels[result])
              #  if label='Smiling'
                
                cv2.imshow('Face', frame_clone)


# Display the result
        


# Release the camera and close all open windows
camera.release()
cv2.destroyAllWindows()






