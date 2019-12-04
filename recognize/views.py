from django.shortcuts import render, redirect
import cv2
import numpy as np
import os 
from record.models import Face
from record.views import names
from datetime import datetime


# Create your views here.

def recognize(request):
    face = list(set(Face.objects.all().values_list('name', flat=True)))    
    face.insert(0,'Unknown')

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trained_model/trainer.yml')
    cascade_path = 'D:\\Users\\gshenoy\\Desktop\\Accounts and Innovation\\POC\\Facial Recognition\\Marcel model 2\\haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    #Getting data from DB
    font = cv2.FONT_HERSHEY_SIMPLEX
    id = 0
    cam = cv2.VideoCapture(0)
    cam.set(3, 640) # set video widht
    cam.set(4, 480) # set video height
    # Define min window size to be recognized as a face
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)
    start_time = 0
    
    while True:
        _, img =cam.read()
        #img = cv2.flip(img, -1) # Flip vertically
        
        
        time_elapsed = datetime.now().second - start_time
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,scaleFactor = 1.3,minNeighbors = 5,minSize = (int(minW), int(minH)))
        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
            # If confidence is less them 100 ==> "0" : perfect match 
            if (confidence < 100):
                person = face[id]
                confidence = "  {0}%".format(round(100 - confidence))
                
            else:
                person = face[0]
                break
            cv2.putText(img, person, (x+5,y-5),font,1,(255,255,255),2)    
        cv2.imshow('camera',img)
        k = cv2.waitKey(10) & 0xff
        if k == 27 or time_elapsed > 20:
            break
    cam.release()
    cv2.destroyAllWindows()   
    return render(request, 'recognize/play.html', {'names':person}) 


    


#Get all Names in a List from DB - Done
#Show it in HTML - Done
# Now design the flow as per DB - Done
# Improve Accuracy of model - try VGG and Facenet
# Beautify page - Done Partiially
# Automatic closure of Camera - Done using datetime.now()

#Research on connecting to JS for video streaming on server
"""
1. Currently We are able to run application in localhost
2. Record -> Records the faces to disk and names to the Postgres DB
3. Recognize -> Runs the camera to recognize the person based on DB
4. For this code to run on server, it has to access client's camera 
    since server has no camera. 
5. VideoCapture should get an input of camera from client side and
     and then run code and show the output back on client camera
6. Clone this project in localhost and then modify

"""