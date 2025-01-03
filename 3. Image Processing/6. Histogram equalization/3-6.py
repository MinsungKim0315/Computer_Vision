import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('Images/blurydog.jpg') 

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
plt.imshow(gray, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.show()

h = cv.calcHist([gray], [0], None, [256], [0, 256])
plt.plot(h, color='r', linewidth=1), plt.show()

equal = cv.equalizeHist(gray)  # Equalize histogram
plt.imshow(equal, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.show()

h = cv.calcHist([equal], [0], None, [256], [0, 256])  # Modify original image using equalized histogram
plt.plot(h, color='r', linewidth=1), plt.show()
