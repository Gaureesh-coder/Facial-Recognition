from django.shortcuts import render, redirect
import os
import cv2
from PIL import Image
from .models import Face
from .train import TRAIN
import numpy as np

names = ['Unknown']


path = 'D:\\Users\\gshenoy\\Desktop\\Accounts and Innovation\\POC\\Facial Recognition\\Marcel model 2\\haarcascade_frontalface_default.xml'
detector = cv2.CascadeClassifier(path)

def record1(request):
    return render(request, 'record/homerecord.html')

def record(request):
    
    face_name = names
    name = request.POST['name']
    face_name.append(name)
    face_id = len(face_name) - 1
    cam = cv2.VideoCapture(0)
    count = 0
    while(True):
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
            count += 1
            cv2.imwrite("dataset/User." + str(face_id) + '.' +  str(count) + ".jpg", gray[y:y+h,x:x+w])
            cv2.imshow('image',img)
        k = cv2.waitKey(100) & 0xff
        if k == 27:
            break
        elif count >= 30:
                break
    cam.release()
    cv2.destroyAllWindows()

    return render(request, 'record/recorddetail.html', {'names': face_name, 'package': count, 'face_id': face_id})
"""
# -----------Below Code is trying to store images in DB ------------------------ #

"""

def record2(request):
    return render(request,'record/recorddb.html')

def record3(request):
    
    #face_name = names
    name = request.POST['name']
    names.append(name)
    
    face_id = len(names) - 1
    cam = cv2.VideoCapture(0)
    count = 0
    while(True):
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
            count += 1
            img_path = "dataset/User." + str(face_id) + '.' + str(count) + ".jpg"
            cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
            face = Face(name=name, face_id = face_id, face_image = img_path)
            face.save()
            cv2.imshow('image',img)
        k = cv2.waitKey(100) & 0xff
        if k == 27:
            break
        elif count >= 30:
                break
    cam.release()
    cv2.destroyAllWindows()

    face_train = TRAIN()
    faces,ids = face_train.getImagesAndLabels(face_train.path)
    face_train.recognizer.train(faces, np.array(ids))
    face_train.recognizer.write('trained_model/trainer.yml') 
    faces_trained = len(np.unique(ids))

    
   
    return render(request, 'record/detaildb.html', {'names': names, 'images': count, 'face_id': face_id, 'trained_faces':faces_trained, 'ids':ids })


#Able to run camera on Django - Able to capture face and store in local system
#Flow of record is correct and working
#Issues: 

#Create a Post request to DB of name, Id and Photo - Created
#Create separate views.py with DB request model and test - 
#We save Media files (images) to disk and reference of that to the DB

#The working model, put it in a separate icon on the Home page with Camera icon
#Auto run train.py - Done

#Connect recognizer
#Compare VGG, Facenet models with this - make same architecture 
#Study Dialogflow and Lex 
#Study on cv2.VideoCapture(1)


