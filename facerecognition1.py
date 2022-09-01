import cv2
import numpy as np
face_classifier=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
def face_extractor(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face = face_classifier.detectMultiScale(gray, 1.3, 5)
    if face is None:
        return None
    for(x,y,w,h) in face:
        
        croppedimg=img[y:y+h+50,x:x+w+50]
        return croppedimg
cap=cv2.VideoCapture(0)
count=0
while (cap.isOpened()):
    ret,frame=cap.read()
    if face_extractor(frame) is not None:
        faces=cv2.resize(face_extractor(frame),(300,300))
        count+=1
        faces=cv2.cvtColor(faces,cv2.COLOR_BGR2GRAY)
        file_name='D:/dataset/'+str(count)+'.jpg'
        cv2.imwrite(file_name,faces)
        cv2.imshow('FaceCropper',faces)
    else:
        print("Face not found")
    if(cv2.waitKey(1)==13 or count==100 ):
        break
cap.release()
cv2.destroyAllWindows()
print("Your Dataset is collected now")

