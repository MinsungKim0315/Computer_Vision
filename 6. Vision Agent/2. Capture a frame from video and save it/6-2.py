from PyQt5.QtWidgets import *
import sys
import cv2 as cv

class Video(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Capture frame from video')
        self.setGeometry(200, 200, 500, 100)
        
        videoButton = QPushButton('open video', self)
        captureButton = QPushButton('capture frame', self)
        saveButton = QPushButton('save', self)
        quitButton = QPushButton('exit', self)
        
        videoButton.setGeometry(10, 10, 100, 30)
        captureButton.setGeometry(110, 10, 100, 30)
        saveButton.setGeometry(210, 10, 100, 30)
        quitButton.setGeometry(310, 10, 100, 30)
        
        videoButton.clicked.connect(self.videoFunction)
        captureButton.clicked.connect(self.captureFunction)
        saveButton.clicked.connect(self.saveFunction)
        quitButton.clicked.connect(self.quitFunction)
        
    def videoFunction(self):
        self.cap = cv.VideoCapture(0, cv.CAP_DSHOW)
        if not self.cap.isOpened():self.close()
        while True:
            ret, self.frame = self.cap.read()   # uses self because cap is used in quitFunction and frame is used in captureFunction
            if not ret: break
            cv.imshow('video display', self.frame)
            cv.waitKey(1)

    def captureFunction(self):
        self.capturedFrame = self.frame
        cv.imshow('Captured frame', self.capturedFrame)
    
    def saveFunction(self):
        fname = QFileDialog.getSaveFileName(self, 'save file', './')
        cv.imwrite(fname[0], self.capturedFrame)
    
    def quitFunction(self):
        self.cap.release()
        cv.destroyAllWindows()
        self.close()
        
app = QApplication(sys.argv)
win = Video()
win.show()
app.exec_()
