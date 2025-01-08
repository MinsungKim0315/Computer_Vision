import cv2 as cv

img = cv.imread('Images/bus_1713.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()
# sift = cv.SIFT_create(nfeatures=00, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
kp, des = sift.detectAndCompute(gray, None)
# kp = sift.detect(gray, None)    # Key point
# des = sift.compute(gray, None)  # Descriptor

gray = cv.drawKeypoints(gray, kp, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imshow('sift', gray)

k = cv.waitKey()
cv.destroyAllWindows()

# center: key point, radius: scale, line in the circle: dominant orientation θ
