import cv2 as cv
import numpy as np

img = cv.imread(r'Images\apples.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

apples = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 45, param1 = 150, param2 = 20, minRadius=10, maxRadius=25)

for i in apples[0]:
    cv.circle(img, (int(i[0]), int(i[1])), int(i[2]), (255, 0, 0), 2)

    
cv.imshow('Apple detection', img)

cv.waitKey()
cv.destroyAllWindows()
