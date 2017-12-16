import cv2
import os
import numpy as np

# opencv_createsamples -info info.lst -num 14229 -w 48 -h 48 -vec positive.vec

# create data file 
# or will error "Parameters can not be written, because file data/params.xml can not be opened."

# opencv_traincascade -data data -vec positive.vec -bg bg.txt -numPos 12000 -numNeg 202 -numStages 20 -w 48 -h 48

# opencv_traincascade -data data -vec positive.vec -bg bg.txt -numPos 12000 -numNeg 202 -numStages 20 -w 48 -h 48

# ----------------------------------------------------------------------------------------------
# use video

# beautiful_face_cascade = cv2.CascadeClassifier('C:\\Code\\BeautifulGirls\\BeautifulFacaCascade.xml')

# cap = cv2.VideoCapture(0)

# while 1:
#     ret, img = cap.read()

#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#     beautiful_face = beautiful_face_cascade.detectMultiScale(gray, 1.3, 5)
#     # 这里参数可改成 5
#     # detectMultiScale()
#     # https://docs.opencv.org/2.4/modules/objdetect/doc/cascade_classification.html
#     # minSize – Minimum possible object size. Objects smaller than that are ignored.
#     # maxSize – Maximum possible object size. Objects larger than that are ignored.
    
#     for (x,y,w,h) in beautiful_face:
#         cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)

#     cv2.imshow('img',img)
#     k = cv2.waitKey(30) & 0xff
#     if k == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()

# ---------------------------------------------------------------------------------------
# use image

BeautifulFacaCascade = cv2.CascadeClassifier('C:\\Code\\BeautifulGirls\\BeautifulFacaCascade.xml')
img = cv2.imread('0-0.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

BeautifulFaca = BeautifulFacaCascade.detectMultiScale(gray, 1.5, 5)
for (x,y,w,h) in BeautifulFaca:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()