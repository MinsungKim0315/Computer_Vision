import cv2 as cv
import sys

cap = cv.VideoCapture(0, cv.CAP_DSHOW)  # Try to connect with webcam

if not cap.isOpened():
    sys.exit('Fail to connect camera')

while True:
    ret, frame = cap.read()  # Get frames
    
    if not ret:
        print('Exit loop due to failure of capturing frame')
        break
    
    cv.imshow('Video display', frame)
    
    key = cv.waitKey(1)
    if key == ord('q'):
        break
    
cap.release()  # Disconnect with webcam
cv.destroyAllWindows()
