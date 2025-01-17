from PyQt5.QtWidgets import *
import cv2 as cv
import numpy as np
import winsound
import sys

class Panorama(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Panorama image')
        self.setGeometry(200, 200, 700, 200)
        
        collectButton = QPushButton('collect image', self)
        self.showButton = QPushButton('view image', self)   # have self in order to allow access from other functions
        self.stitchButton = QPushButton('stitch', self)
        self.saveButton = QPushButton('save', self)
        quitButton = QPushButton('exit', self)
        self.label=QLabel('Welcome to Panomara maker!', self)
        
        collectButton.setGeometry(10, 25, 100, 30)
        self.showButton.setGeometry(110, 25, 100, 30)
        self.stitchButton.setGeometry(210, 25, 100, 30)
        self.saveButton.setGeometry(310, 25, 100, 30)
        quitButton.setGeometry(450, 25, 100, 30)
        self.label.setGeometry(10, 70, 600, 170)
        
        self.showButton.setEnabled(False)
        self.stitchButton.setEnabled(False)
        self.saveButton.setEnabled(False)
        
        collectButton.clicked.connect(self.collectFunction)
        self.showButton.clicked.connect(self.showFunction)
        self.stitchButton.clicked.connect(self.stitchFunction)
        self.saveButton.clicked.connect(self.saveFunction)
        quitButton.clicked.connect(self.quitFunction)
    
    def collectFunction(self):
        self.showButton.setEnabled(False)
        self.stitchButton.setEnabled(False)
        self.saveButton.setEnabled(False)
        self.label.setText('press "c" multiple times to collect frames from video and press "q" to end video')
        
        self.cap = cv.VideoCapture(0, cv.CAP_DSHOW)
        if not self.cap.isOpened(): sys.exit('Fail to connect Web Cam')
        
        self.imgs = []
        while True:
            ret, frame = self.cap.read()
            if not ret: break
            
            cv.imshow('video display', frame)
            
            key = cv.waitKey(1)
            if key == ord('c'):
                self.imgs.append(frame)
            elif key == ord('q'):
                self.cap.release()
                cv.destroyWindow('video display')
                break
        
        if len(self.imgs)>=2:
            self.showButton.setEnabled(True)
            self.stitchButton.setEnabled(True)
            self.saveButton.setEnabled(True)
        else:
            self.label.setText('You need to collect at least two frames!')
    
    def showFunction(self):
        self.label.setText(str(len(self.imgs))+'collected frames')
        # stack = cv.resize(self.imgs[0], dsize=(0,0), fx=0.25, fy=0.25)
        # for i in range(1, len(self.imgs)):
        #     stack=np.hstack((stack, cv.resize(self.imgs[i], dsize=(0,0), fx=0.25, fy=0.25)))
        stack = np.hstack([cv.resize(img, dsize=(0, 0), fx=0.25, fy=0.25) for img in self.imgs])
        cv.imshow('Image collection', stack)

    def stitchFunction(self):
        stitcher = cv.Stitcher_create()
        status, self.img_stitched = stitcher.stitch(self.imgs)
        if status == cv.STITCHER_OK:
            cv.imshow('Image stitched panorama', self.img_stitched)
        else:
            winsound.Beep(3000, 500)
            self.label.setText('Failed to create panorama. Try again')
    
    def saveFunction(self):
        fname = QFileDialog.getSaveFileName(self, 'save file', './')
        cv.imwrite(fname[0], self.img_stitched)
    
    def quitFunction(self):
        self.cap.release()
        cv.destroyAllWindows()
        self.close()

app = QApplication(sys.argv)
win = Panorama()
win.show()
app.exec_()
