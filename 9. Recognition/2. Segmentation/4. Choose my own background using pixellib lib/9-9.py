import cv2 as cv
import numpy as np
from PyQt5.QtWidgets import *
import sys
from pixellib.tune_bg import alter_bg

class VideoSpecialEffect(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Choose my own background')
        self.setGeometry(200, 200, 400, 100)
        
        videoButton = QPushButton('Start video', self)
        self.pickCombo = QComboBox(self)
        self.pickCombo.addItems(['Original', 'blur(min)', 'blur(mid)', 'blur(max)', 'Red', 'Green', 'Blue'])
        quitButton = QPushButton('exit', self)
        
        videoButton.setGeometry(10, 10, 140, 30)
        self.pickCombo.setGeometry(150, 10, 110, 30)
        quitButton.setGeometry(280, 10, 100, 30)
        
        videoButton.clicked.connect(self.videoSpecialEffectFunction)
        quitButton.clicked.connect(self.quitFunction)
        
    def videoSpecialEffectFunction(self):
        self.cap = cv.VideoCapture(0, cv.CAP_DSHOW)
        if not self.cap.isOpened(): sys.exit('failed to connect camera')
        
        while True:
            ret, frame = self.cap.read()
            if not ret: break
            
            pick_effect = self.pickCombo.currentIndex()
            if pick_effect == 0:
                special_img = frame
            elif pick_effect == 1:
                special_img = change_bg.blur_frame(frame, low=True, detect='person')
            elif pick_effect == 2:
                special_img = change_bg.blur_frame(frame, moderate=True, detect='person')
            elif pick_effect == 3:
                special_img = change_bg.blur_frame(frame, extreme=True, detect='person')
            elif pick_effect == 4:
                special_img = change_bg.color_frame(frame, colors=(255,0,0), detect='person')
            elif pick_effect == 5:
                special_img = change_bg.color_frame(frame, colors=(0,255,0), detect='person')
            elif pick_effect == 6:
                special_img = change_bg.color_frame(frame, colors=(0,0,255), detect='person')
            
            cv.imshow('Special effect', special_img)
            cv.waitKey(1)
    
    def quitFunction(self):
        self.cap.release()
        cv.destroyAllWindows()
        self.close()

change_bg = alter_bg(model_type="pb")
change_bg.load_pascalvoc_model('xception_pascalvoc.pb')

app = QApplication(sys.argv)
win = VideoSpecialEffect()
win.show()
app.exec_()
