import cv2 as cv
import numpy as np
from PyQt5.QtWidgets import *
import sys

class SpecialEffect(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('image special effect')
        self.setGeometry(200, 200, 800, 200)
        
        pictureButton = QPushButton('read image', self)
        embossButton = QPushButton('embossing', self)
        cartoonButton = QPushButton('cartoon', self)
        sketchButton = QPushButton('pencil sketch', self)
        oilButton = QPushButton('oil', self)
        saveButton = QPushButton('save', self)
        self.pickCombo = QComboBox(self)    # Create a combo box
        self.pickCombo.addItems(['embossing', 'cartoon', 'pencil sketch(gray)', 'penceil sketch(color)', 'oil']) # options in the combo box
        quitButton = QPushButton('exit', self)
        self.label = QLabel('Welcome to Special Effect!', self)
        
        pictureButton.setGeometry(10, 10, 100, 30)
        embossButton.setGeometry(110, 10, 100, 30)
        cartoonButton.setGeometry(210, 10, 100, 30)
        sketchButton.setGeometry(310, 10, 100, 30)
        oilButton.setGeometry(410, 10, 100, 30)
        saveButton.setGeometry(510, 10, 100, 30)
        self.pickCombo.setGeometry(510, 40, 110, 30)
        quitButton.setGeometry(620, 10, 100, 30)
        self.label.setGeometry(10, 40, 500, 170)
        
        pictureButton.clicked.connect(self.pictureOpenFunction)
        embossButton.clicked.connect(self.embossFunction)
        cartoonButton.clicked.connect(self.cartoonFunction)
        sketchButton.clicked.connect(self.sketchFunction)
        oilButton.clicked.connect(self.oilFunction)
        saveButton.clicked.connect(self.saveFunction)
        quitButton.clicked.connect(self.quitFunction)
    
    def pictureOpenFunction(self):
        fname = QFileDialog.getOpenFileName(self, 'get image', './')
        self.img = cv.imread(fname[0])
        if self.img is None: sys.exit('cannot find file')
        
        cv.imshow('Original', self.img)
        
    def embossFunction(self):
        femboss = np.array([[-1.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 1.0]])
        
        gray = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        gray16 = np.int16(gray)
        self.emboss = np.uint8(np.clip(cv.filter2D(gray16, -1, femboss)+128, 0, 255))
        
        cv.imshow('Emboss', self.emboss)
        
    def cartoonFunction(self):
        self.cartoon = cv.stylization(self.img, sigma_s=60, sigma_r=0.45)
        
        cv.imshow('Cartoon', self.cartoon)
        
    def sketchFunction(self):
        # pencilSketch function returns two images; a gray one and a color one
        self.sketch_gray, self.sketch_color = cv.pencilSketch(self.img, sigma_s=60, sigma_r=0.07, shade_factor=0.02)
        cv.imshow('Pencil sketch(gray)', self.sketch_gray)
        cv.imshow('Pencil sketch(color)', self.sketch_color)
    
    def oilFunction(self):
        self.oil = cv.xphoto.oilPainting(self.img, 10, 1, cv.COLOR_BGR2Lab)
        cv.imshow('Oil painting', self.oil)
        
    def saveFunction(self):
        fname,_ = QFileDialog.getSaveFileName(self, 'save file', './', "Images (*.png *.jpg *.bmp)")
        i = self.pickCombo.currentIndex()
        try:
            if i == 0 and hasattr(self, 'emboss') and self.emboss is not None:
                cv.imwrite(fname, self.emboss)
            elif i == 1 and hasattr(self, 'cartoon') and self.cartoon is not None:
                cv.imwrite(fname, self.cartoon)
            elif i == 2 and hasattr(self, 'sketch_gray') and self.sketch_gray is not None:
                cv.imwrite(fname, self.sketch_gray)
            elif i == 3 and hasattr(self, 'sketch_color') and self.sketch_color is not None:
                cv.imwrite(fname, self.sketch_color)
            elif i == 4 and hasattr(self, 'oil') and self.oil is not None:
                cv.imwrite(fname, self.oil)
            else:
                QMessageBox.warning(self, 'Error', 'The selected effect is not available or has not been applied.')
        except Exception as e:
            QMessageBox.critical(self, 'Error', f"Failed to save the image: {e}")
    
    def quitFunction(self):
        cv.destroyAllWindows()
        self.close()
        
app = QApplication(sys.argv)
win = SpecialEffect()
win.show()
app.exec_()
