import cv2 as cv
import numpy as np

img = cv.imread('Images\soccer.png')
img = cv.resize(img, dsize = (0,0), fx=0.6, fy=0.6)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.putText(gray, 'soccer', (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
cv.imshow('Original', gray)

# Applying smoothing to image, the 0.0 will automatically calculate the σ depending on the filter size
smooth = np.hstack((cv.GaussianBlur(gray, (5, 5), 0.0), cv.GaussianBlur(gray, (9, 9), 0.0), cv.GaussianBlur(gray, (15, 15), 0.0)))
cv.imshow('Smooth', smooth)

# Applying embossing and compare results
femboss = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

gray16 = np.int16(gray)
emboss = np.uint8(np.clip(cv.filter2D(gray16, -1, femboss)+128, 0, 225))    # Proper way
emboss_bad = np.uint8(cv.filter2D(gray16, -1, femboss)+128)                 # Exclude np.clip
emboss_worse = cv.filter2D(gray16, -1, femboss)                             # Cause overflow

cv.imshow('Emboss', emboss)
cv.imshow('Emboss_bad', emboss_bad)
cv.imshow('Emboss_worse', emboss_worse)

cv.waitKey()
cv.destroyAllWindows()
