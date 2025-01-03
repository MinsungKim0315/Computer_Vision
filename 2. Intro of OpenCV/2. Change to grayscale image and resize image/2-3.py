import cv2 as cv
import sys

img = cv.imread('soccer.png')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # Change the image to gray scale
gray_small = cv.resize(gray, dsize=(0, 0), fx=0.5, fy=0.5)  # Control the size of the image

cv.imwrite('soccer_gray.png', gray)
cv.imwrite('soccer_gray_small.png', gray_small)

cv.imshow('Color image', img)
cv.imshow('Gray image', gray)
cv.imshow('Gray small image', gray_small)

cv.waitKey()
cv.destroyAllWindows()
