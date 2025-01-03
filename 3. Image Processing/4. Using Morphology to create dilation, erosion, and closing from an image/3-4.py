import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sys

img = cv.imread('Images/soccer.png', cv.IMREAD_UNCHANGED)
if img is None:
    print("Error: Image not loaded. Check the file path.")
    sys.exit()
    
t, bin_img = cv.threshold(img[:, :, 2], 0, 225, cv.THRESH_BINARY+cv.THRESH_OTSU)
plt.imshow(bin_img, cmap='gray'), plt.xticks([]), plt.yticks([])
plt.show()

b = bin_img[bin_img.shape[0]//2:bin_img.shape[0], 0:bin_img.shape[0]//2+1]
plt.imshow(b, cmap = 'gray'), plt.xticks([]), plt.yticks([])

se = np.uint8([[0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0]])    # structing element for morphology

b_dilation = cv.dilate(b, se, iterations=1)
plt.imshow(b_dilation, cmap='gray'), plt.xticks([]), plt.yticks([])
plt.show()

b_erosion = cv.erode(b, se, iterations=1)
plt.imshow(b_erosion, cmap='gray'), plt.xticks([]), plt.yticks([])
plt.show()

b_closing = cv.erode(cv.dilate(b, se, iterations=1), se, iterations=1)  # Closing: apply erosion to a dilated image
plt.imshow(b_closing, cmap='gray'), plt.xticks([]), plt.yticks([])
plt.show()
