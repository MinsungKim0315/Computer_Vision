import cv2 as cv
import numpy as np
import sys
from PyQt5.QtWidgets import *

class GrabCut(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('GrabCut')
        self.setGeometry(200, 200, 700, 200)
        
        fileButton = QPushButton('file', self)
        paintButton = QPushButton('paint(grab)', self)
        cutButton = QPushButton('cut', self)
        incButton = QPushButton('+', self)
        decButton = QPushButton('-', self)
        saveButton = QPushButton('save', self)
        quitButton = QPushButton('exit', self)
        
        fileButton.setGeometry(10, 10, 100, 30)
        paintButton.setGeometry(110, 10, 100, 30)
        cutButton.setGeometry(210, 10, 100, 30)
        incButton.setGeometry(310, 10, 50, 30)
        decButton.setGeometry(360, 10, 50, 30)
        saveButton.setGeometry(410, 10, 100, 30)
        quitButton.setGeometry(510, 10, 100, 30)
        
        fileButton.clicked.connect(self.fileOpenFunction)
        paintButton.clicked.connect(self.paintFunction)
        cutButton.clicked.connect(self.cutFunction)
        incButton.clicked.connect(self.incFunction)
        decButton.clicked.connect(self.decFunction)
        saveButton.clicked.connect(self.saveFunction)
        quitButton.clicked.connect(self.quitFunction)
        
        self.BrushSiz = 5
        self.Lcolor, self.Rcolor = (255, 0, 0), (0, 0, 255)
    
    def fileOpenFunction(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', './')
        self.img = cv.imread(fname[0])
        if self.img is None: sys.exit('Cannot find file')
        
        self.img_show = np.copy(self.img)
        cv.imshow('Painting', self.img_show)    # Image for display
        
        self.mask = np.zeros((self.img.shape[0], self.img.shape[1]), np.uint8)  # save user's painting info
        self.mask[:,:]=cv.GC_PR_BGD             # Initialize all pixel to 'BGD' (background)
        
    def paintFunction(self):
        cv.setMouseCallback('Painting', self.painting)  # if something happen in window called 'Painting', then call def painting
        
    def painting(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            cv.circle(self.img_show, (x, y), self.BrushSiz, self.Lcolor, -1)    # Blue
            cv.circle(self.mask, (x, y), self.BrushSiz, cv.GC_FGD, -1)          # Set to FFD (foreground)
        elif event == cv.EVENT_RBUTTONDOWN:
            cv.circle(self.img_show, (x, y), self.BrushSiz, self.Rcolor, -1)    # Red
            cv.circle(self.mask, (x, y), self.BrushSiz, cv.GC_BGD, -1)          # Set to BGD
        elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:
            cv.circle(self.img_show, (x, y), self.BrushSiz, self.Lcolor, -1)
            cv.circle(self.mask, (x, y), self.BrushSiz, cv.GC_FGD, -1)
        elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_RBUTTON:
            cv.circle(self.img_show, (x, y), self.BrushSiz, self.Rcolor, -1)
            cv.circle(self.mask, (x, y), self.BrushSiz, cv.GC_BGD, -1)
        
        cv.imshow('Painting', self.img_show)
    
    def cutFunction(self):
        background = np.zeros((1, 65), np.float64)
        foreground = np.zeros((1, 65), np.float64)
        cv.grabCut(self.img, self.mask, None, background, foreground, 5, cv.GC_INIT_WITH_MASK)
        mask2 = np.where((self.mask == 2)|(self.mask == 0), 0, 1).astype('uint8')
        self.grabImg = self.img*mask2[:,:,np.newaxis]
        cv.imshow('Scissoring', self.grabImg)
    
    def incFunction(self):
        self.BrushSiz = min(20, self.BrushSiz+1)
    
    def decFunction(self):
        self.BrushSiz = max(1, self.BrushSiz-1)
    
    def saveFunction(self):
        fname = QFileDialog.getSaveFileName(self, 'save file', './')
    
    def quitFunction(self):
        cv.destroyAllWindows()
        self.close()
    
app = QApplication(sys.argv)
win = GrabCut()
win.show()
app.exec_()
