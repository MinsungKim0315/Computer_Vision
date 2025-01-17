import cv2 as cv
import numpy as np
import tensorflow as tf
import winsound
import pickle
import sys
from PyQt5.QtWidgets import *

cnn = tf.keras.models.load_model('\cnn_for_stanford_dogs.h5')
dog_species = pickle.load(open('dog_species_names.txt', 'rb'))

class DogSpeciesRecognition(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Recognize Species')
        self.setGeometry(200,200,700,100)
        
        fileButton = QPushButton('open dog image', self)
        recognitionButton = QPushButton('recognize species', self)
        quitButton = QPushButton('exit', self)
        
        fileButton.setGeometry(10, 10, 100, 30)
        recognitionButton.setGeometry(110, 10, 100, 30)
        quitButton.setGeometry(510, 10, 100, 30)
        
        fileButton.clicked.connect(self.pictureOpenFunction)
        recognitionButton.clicked.connect(self.recognitionFunction)
        quitButton.clicked.connect(self.quitFunction)
        
    def pictureOpenFunction(self):
        fname = QFileDialog.getOpenFileName(self,'read dog image', './')
        self.img = cv.imread(fname[0])
        if self.img is None: sys.exit()
        
        cv.imshow('Doge Image', self.img)
        
    def recognitionFunction(self):
        x = np.reshape(cv.resize(self.img, (224,224)), (1, 224,224,3))
        res = cnn.predict(x)[0]
        top5 = np.argsort(-res)[:5]
        top5_dog_species_names = [dog_species[i] for i in top5]
        for i in range(5):
            prob = '('+str(res[top5[i]])+')'
            name = str(top5_dog_species_names[i]).split('-')[1]
            cv.putText(self.img, prob+name, (10,100+i*30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv.imshow('Dog Image', self.img)
        winsound.Beep(1000, 500)
    
    def quitFunction(self):
        cv.destroyAllWindows()
        self.close()

app = QApplication(sys.argv)
win = DogSpeciesRecognition()
win.show()
app.exec_()
