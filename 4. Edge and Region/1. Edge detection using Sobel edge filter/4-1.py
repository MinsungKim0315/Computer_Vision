import cv2 as cv

img = cv.imread(r'C:\Users\idkms\.spyder-py3\Computer Vision\Images\soccer.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

grad_x = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3)   # intensifies vertical edges
grad_y = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)   # intensifies horizontal edges

sobel_x = cv.convertScaleAbs(grad_x)
sobel_y = cv.convertScaleAbs(grad_y)

edge_strength = cv.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)   # intensifies edges with high contrast

cv.imshow('Original', gray)
cv.imshow('sobelx', sobel_x)
cv.imshow('sobely', sobel_y)
cv.imshow('edge strength', edge_strength)

cv.waitKey()
cv.destroyAllWindows()
