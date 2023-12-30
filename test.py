import cv2
import numpy as np
import imutils
from keras.preprocessing.image import img_to_array

from imutils import paths
from keras.models import load_model



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
        model = load_model('model.h5')
        face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
        frame_clone = frame.copy()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
        print(faces)
        for (x, y, width, height) in faces:   # Get the region of interest
                cv2.rectangle(frame,(x,y),(x+width,y+height),(0,255,0),2)
                #qcv2.imshow('frame',frame)
                roi = gray[y:y+height,x:x+width] # GETS THE FACE (REGION OF INTEREST) 
             
                roi = cv2.resize(roi, (28,28))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                
                (notSmiling, Smiling) = model.predict(roi)[0]
              #  if label='Smiling'
                label = 'Smiling' if Smiling > notSmiling else "Not Smiling"
                if label == 'Smiling':
                        cv2.putText(frame_clone, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
                        cv2.rectangle(frame_clone, (x, y), (x + width, y + height), (0, 255, 0), 2)
                else:
                        cv2.putText(frame_clone, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                        cv2.rectangle(frame_clone, (x, y), (x + width, y + height), (0, 0, 255), 2)
                cv2.imshow('Face', frame_clone)

                #print(label)

# Display the result
        


# Release the camera and close all open windows
camera.release()
cv2.destroyAllWindows()