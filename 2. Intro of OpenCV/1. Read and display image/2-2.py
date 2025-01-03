import cv2 as cv
import sys

img = cv.imread('soccer.png')  # Read the image file

if img is None:
  sys.exit('Cannot find the file')

cv.imshow('Image display', img)  # Show the image in a window

cv.waitKey()  # Wait until there is a keyboard input
cv.destroyAllWindows()  # Shutdown all open windows
