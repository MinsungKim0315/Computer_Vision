import cv2 as cv
import sys

img = cv.imread('Images/makima.png')
img = cv.resize(img, dsize=(0, 0), fx=0.3, fy=0.3)

BrushSize = 5
Lcolor, Rcolor = (255, 0, 0), (0, 0, 255)

def painting(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(img, (x, y), BrushSize, Lcolor, -1)
    elif event == cv.EVENT_RBUTTONDOWN:
        cv.circle(img, (x, y), BrushSize, Rcolor, -1)
    elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:
        cv.circle(img, (x, y), BrushSize, Lcolor, -1)
    elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_RBUTTON:
        cv.circle(img, (x, y), BrushSize, Rcolor, -1)
    
    cv.imshow('Painting', img)
    
cv.namedWindow('Painting')
cv.imshow('Painting', img)

cv.setMouseCallback('Painting', painting)

while(True):
    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        break
