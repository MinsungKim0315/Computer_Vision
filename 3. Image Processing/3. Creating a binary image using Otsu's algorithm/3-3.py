import cv2 as cv
import sys

img = cv.imread('Images/soccer.png')

t, bin_img = cv.threshold(img[:, :, 0], 0, 225, cv.THRESH_BINARY+cv.THRESH_OTSU)
print('The best threshold = ', t)

cv.imshow('R channel', img[:, :, 0])
cv.imshow('R channel Binarization', bin_img)

cv.waitKey()
cv.destroyAllWindows()
