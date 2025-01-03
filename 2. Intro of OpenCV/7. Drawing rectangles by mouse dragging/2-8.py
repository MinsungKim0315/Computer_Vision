import cv2 as cv

img = cv.imread('Images/cat and dog.jpg')

def draw(event, x, y, flags, param):
    global ix, iy
    
    if event == cv.EVENT_LBUTTONDOWN:
        ix, iy = x, y
    elif event == cv.EVENT_LBUTTONUP:
        cv.rectangle(img, (ix, iy), (x, y), (0, 0, 225), 2)
    elif event == cv.EVENT_RBUTTONDOWN:
        ix, iy = x, y
    elif event == cv.EVENT_RBUTTONUP:
        cv.rectangle(img, (ix, iy), (x, y), (225, 0, 0), 2)
        
    cv.imshow('Drawing', img)
    
cv.namedWindow('Drawing')
cv.imshow('Drawing', img)

cv.setMouseCallback('Drawing', draw)

while True:
    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        break
