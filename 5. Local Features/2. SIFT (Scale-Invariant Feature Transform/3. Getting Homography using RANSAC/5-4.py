import cv2 as cv
import numpy as np

img1 = cv.imread('Images/bus_3170.png')[180:360, 420:560]   # Model image
gray1 =cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2 = cv.imread('Images/bus_3305.png')                     # Scene image
gray2 =cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

sift=cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)  # get key point and descriptor from model image
kp2, des2 = sift.detectAndCompute(gray2, None)  # get key point and descriptor from scene image

flann_matcher = cv.DescriptorMatcher_create(cv.DESCRIPTOR_MATCHER_FLANNBASED)
knn_match = flann_matcher.knnMatch(des1, des2, 2)   # get 2 nearest neighbors

T = 0.7
good_match = []
for nearest1, nearest2 in knn_match:
    if (nearest1.distance/nearest2.distance)<T:  # Nearest neighbor distance ratio
        good_match.append(nearest1)

points1 = np.float32([kp1[gm.queryIdx].pt for gm in good_match])    # key points from Model image that are in good_match
points2 = np.float32([kp2[gm.trainIdx].pt for gm in good_match])    # key points from Scene image that are in good_match

H, _=cv.findHomography(points1, points2, cv.RANSAC) # get Homography Matrix using RANSAC algorithm

h1, w1 = img1.shape[0], img1.shape[1]   # figure out size of img1
h2, w2 = img2.shape[0], img2.shape[1]   # figure out size of img2

box1 = np.float32([[0, 0], [0, h1-1], [w1-1, h1-1], [w1-1, 0]]).reshape(4, 1, 2)    # save the coordinates of the 4 corners of img1
box2 = cv.perspectiveTransform(box1, H) # project the 4 coordinates of box1 using H and save the results in box2

img2 = cv.polylines(img2, [np.int32(box2)], True, (0, 255, 0), 8)   # draw box2 in img2

img_match = np.empty((max(h1, h2), w1+w2, 3), dtype=np.uint8)
cv.drawMatches(img1, kp1, img2, kp2, good_match, img_match, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv.imshow('Matches and Homography', img_match)

k=cv.waitKey()  # save the ascii code of the input that closed window
cv.destroyAllWindows()
