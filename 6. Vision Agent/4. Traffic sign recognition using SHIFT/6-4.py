import cv2 as cv
import numpy as np
from PyQt5.QtWidgets import *
import sys
import winsound


class TrafficSign(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Traffic signs')
        self.setGeometry(200, 200, 700, 200)

        signButton = QPushButton('bring sign', self)
        roadButton = QPushButton('upload road', self)
        recognitionButton = QPushButton('recognition', self)
        quitButton = QPushButton('exit', self)
        self.label = QLabel('welcome to traffic sign recognizer', self)

        signButton.setGeometry(10, 10, 100, 30)
        roadButton.setGeometry(110, 10, 100, 30)
        recognitionButton.setGeometry(210, 10, 100, 30)
        quitButton.setGeometry(510, 10, 100, 30)
        self.label.setGeometry(10, 40, 600, 170)

        signButton.clicked.connect(self.signFunction)
        roadButton.clicked.connect(self.roadFunction)
        recognitionButton.clicked.connect(self.recognitionFunction)
        quitButton.clicked.connect(self.quitFunction)

        self.signFiles = [['Images/Stop.jpg', 'Stop'], ['Images/NoUturn.jpg',
                                                        'No U Turn'], ['Images/DeadEnd.jpg', 'Dead End']]
        self.signImgs = []

    def signFunction(self):
        self.label.clear()
        self.label.setText('Register traffic signs')
        
        # bring images in signFiles and save it in fname
        for fname, _ in self.signFiles:
            self.signImgs.append(cv.imread(fname))
            cv.imshow(fname, self.signImgs[-1])

    def roadFunction(self):
        if self.signImgs == []:
            self.label.setText('Register traffic signs first!')
        # make user browse file and choos road image and save is in roadImg
        else:
            fname = QFileDialog.getOpenFileName(self, 'read file', './')
            self.roadImg = cv.imread(fname[0])
            if self.roadImg is None:
                sys.exit('No file')

            cv.imshow('Road scene', self.roadImg)

    def recognitionFunction(self):
        if self.roadImg is None:
            self.label.setText('Upload road scene first!')
        else:
            sift = cv.SIFT_create()

            KD = []
            # get the kp and des from thee signImgs and save it in KD: [[kp1, des1], [kp2, des2], [kp3, des3]]
            for img in self.signImgs:
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                KD.append(sift.detectAndCompute(gray, None))
            
            # get the kp and des from roadImg
            grayRoad = cv.cvtColor(self.roadImg, cv.COLOR_BGR2GRAY)
            road_kp, road_des = sift.detectAndCompute(grayRoad, None)
            
            matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
            
            # extract the kp and des from KD and save it in sign_kp and sign_des
            # match sign_des and road_des one by one
            # find the best match and save it in GM
            # GM has the best match list for each image
            GM = []
            for sign_kp, sign_des in KD:
                knn_match = matcher.knnMatch(sign_des, road_des, 2)
                T = 0.7
                good_match = []
                for nearest1, nearest2 in knn_match:
                    if (nearest1.distance/nearest2.distance)<T:
                        good_match.append(nearest1)
                GM.append(good_match)
            
            # Find the longest match list in GM and save it in best
            best = GM.index(max(GM, key = len))
            
            # If all match list is short then 4, then there is no sign in road that matches
            if len(GM[best])<4:
                self.label.setText('No sign')
            # Find the homography and show it in image
            else:
                sign_kp = KD[best][0]   # get the kp from the best sign image
                good_match = GM[best]   # get the matching info from GM
                
                points1 = np.float32([sign_kp[gm.queryIdx].pt for gm in good_match])
                points2 = np.float32([road_kp[gm.trainIdx].pt for gm in good_match])
                
                H,_ = cv.findHomography(points1, points2, cv.RANSAC)    # get matrix H
                
                h1, w1 = self.signImgs[best].shape[0], self.signImgs[best].shape[1]
                h2, w2 = self.roadImg.shape[0], self.roadImg.shape[1]
                
                box1 = np.float32([[0, 0], [0, h1-1], [w1-1, h1-1], [w1-1, 0]]).reshape(4, 1, 2)
                box2 = cv.perspectiveTransform(box1, H)
                
                self.roadImg = cv.polylines(self.roadImg, [np.int32(box2)], True, (0, 255, 0), 4)
                
                img_match = np.empty((max(h1, h2), w1+w2, 3), dtype=np.uint8)
                cv.drawMatches(self.signImgs[best], sign_kp, self.roadImg, road_kp, good_match, img_match, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                cv.imshow('Matches and Homography', img_match)
                
                self.label.setText(self.signFiles[best][1]+' Watch Out!')
                winsound.Beep(3000, 500)
    
    def quitFunction(self):
        cv.destroyAllWindows()
        self.close()

app = QApplication(sys.argv)
win = TrafficSign()
win.show()
app.exec_()
