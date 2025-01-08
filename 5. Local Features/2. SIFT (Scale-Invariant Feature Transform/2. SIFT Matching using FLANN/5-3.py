import cv2 as cv
import numpy as np
import time

img1 = cv.imread('Images/bus_1713.png')[190:350, 440:560]   # Model image
gray1 =cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2 = cv.imread('Images/bus_3305.png')                     # Scene image
gray2 =cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

sift=cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)  # get key point and descriptor from model image
kp2, des2 = sift.detectAndCompute(gray2, None)  # get key point and descriptor from scene image
print('number of key points: ', len(kp1),',', len(kp2))

start = time.time() # start timer
flann_matcher = cv.DescriptorMatcher_create(cv.DESCRIPTOR_MATCHER_FLANNBASED)
knn_match = flann_matcher.knnMatch(des1, des2, 2)   # get 2 nearest neighbors

T = 0.7
good_match = []
for nearest1, nearest2 in knn_match:
    if(nearest1.distance/nearest2.distance)<T:  # Nearest neighbor distance ratio
        good_match.append(nearest1)
print('Running time for matching: ', time.time()-start) # end timer

img_match = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype = np.uint8) # array for showing model and scene image together
cv.drawMatches(img1, kp1, img2, kp2, good_match, img_match, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)   # draw matches

cv.imshow('Good Matches', img_match)

k=cv.waitKey()
cv.destroyAllWindows()
