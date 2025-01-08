import cv2 as cv
import numpy as np

img = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],dtype=np.float32)

ux = np.array([[-1, 0, 1]])             # horizontal derivative kernel: computes the rate of change along the x-axis
uy = np.array([-1, 0, 1]).transpose()   # vertical derivative kernel: computes the rate of change along the y-axis
k = cv.getGaussianKernel(3, 1)          # 1D Gaussian Kernel (1D array with 3 elements)
g = np.outer(k, k.transpose())          # g = k ⊗ k^T (outer product)

dy = cv.filter2D(img, cv.CV_32F, uy)    # captures vertical intensity changes
dx = cv.filter2D(img, cv.CV_32F, ux)    # captures horizontal intensity changes
dyy = dy*dy
dxx = dx*dx
dyx = dy*dx
gdyy = cv.filter2D(dyy, cv.CV_32F, g)
gdxx = cv.filter2D(dxx, cv.CV_32F, g)
gdyx = cv.filter2D(dyx, cv.CV_32F, g)
C = (gdyy*gdxx - gdyx*gdyx)-0.04*(gdyy+gdxx)*(gdyy+gdxx)

# Non-Maximum Suppression: only the most prominent responses are retained while suppressing weaker or redundant responses
for j in range(1, C.shape[0]-1):    # Iterates over the inner region of the array C, skipping the borders
    for i in range(1, C.shape[1]-1):
        # C[j, i] > 0.1: threshold to ignore weak responses or noise
        # C[j-1:j+2, i-1:i+2]: Extracts a 3x3 neighborhood centered at (j,i)
        # C[j, i]>C[j-1:j+2, i-1:i+2]: Compares the central pixel to all its 8 neighbors
        # sum(sum())==8: counts how many neighbors the central pixel is greater than and if it's exactly 8, the central pixel is strictly greater than all neighbors
        if C[j, i] > 0.1 and sum(sum(C[j, i]>C[j-1:j+2, i-1:i+2])) == 8:
            img[j, i] = 9

np.set_printoptions(precision = 2)
print('dy:\n',dy)
print('dx:\n',dx)
print('dyy:\n',dyy)
print('dxx:\n',dxx)
print('dyx:\n',dyx)
print('gdyy:\n',gdyy)
print('gdxx:\n',gdxx)
print('gdyx:\n',gdyx)
print('C:\n',C)
print('img:\n',img)

popping = np.zeros([160, 160], np.uint8)
for j in range(0, 160):
    for i in range(0, 160):
        popping[j,i] = np.uint8((C[j//16, i//16]+0.06)*700)
        
cv.imshow('Image Display2', popping)
cv.waitKey()
cv.destroyAllWindows()
