
import cv2
import numpy as np
from PIL import Image
import os
from .models import Face

class TRAIN():
    
    path = 'dataset'
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    def __init__(self,):
        cascade_path = 'D:\\Users\\gshenoy\\Desktop\\Accounts and Innovation\\POC\\Facial Recognition\\Marcel model 2\\haarcascade_frontalface_default.xml'
        self.detector = cv2.CascadeClassifier(cascade_path)
        
    def getImagesAndLabels(self,path):
        
        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
        faceSamples=[]
        ids = []
        for imagePath in imagePaths:
            PIL_img = Image.open(imagePath).convert('L') # grayscale
            img_numpy = np.array(PIL_img,'uint8')
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = self.detector.detectMultiScale(img_numpy)
            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                ids.append(id)
        return faceSamples,ids
