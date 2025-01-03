import cv2 as cv
import numpy as np
import sys

cap = cv.VideoCapture(0, cv.CAP_DSHOW)  # Try to connect camera

if not cap.isOpened():
    sys.exit('Failed to Connect Camera')

frames = []
while True:
    ret, frame = cap.read() # Get frame from video
    
    if not ret:
        print('Failed to bring frame info')
        break
    cv.imshow('Video Display', frame)
    key = cv.waitKey(1)
    if key == ord('c'): # if 'c' is clicked, then save frame in the list
        frames.append(frame)
    if key == ord('q'): # if 'q' is clicked, then escape loop
        break
    

cap.release()
cv.destroyAllWindows()

if len(frames) > 0:
    imgs = frames[0]

    for i in range(1, min(3, len(frames))):
        imgs = np.hstack((imgs, frames[i]))  # Stitch maximum three collected images from the video
    
    imgs_small = cv.resize(imgs, dsize=(0, 0), fx=0.5, fy=0.5)
    cv.imshow('Collected images', imgs_small)
    
    cv.waitKey()
    cv.destroyAllWindows()
