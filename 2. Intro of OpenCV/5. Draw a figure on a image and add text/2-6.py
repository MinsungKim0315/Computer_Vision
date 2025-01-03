import cv2 as cv

img = cv.imread('Images/cat and dog.jpg')

cv.rectangle(img, (350, 30), (500, 100), (0, 0, 255), 2)    # (img, position, size, collor, thickness)
cv.putText(img, 'SLAP!!', (350, 24), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3) # (img, word, position, font, size, collor, thickness)

cv.imshow('Draw', img)

cv.waitKey()
cv.destroyAllWindows()
