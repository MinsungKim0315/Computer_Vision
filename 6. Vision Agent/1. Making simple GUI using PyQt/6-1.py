from PyQt5.QtWidgets import *
import sys
import winsound

class BeepSound(QMainWindow):
    def __init__(self):
        super().__init__()  # ensures that base functionality of QMainWindow is initialized before you add custon widgets and functions
        self.setWindowTitle('Make a beep sound')    # name of window
        self.setGeometry(200, 200, 500, 100)    # (200, 200) position of the screen with 500 width and 100 height
        
        # QPushButton: widget for making button, QLabel: widget for making label
        shortBeepButton = QPushButton('short beep', self)   # make a button called 'short beep' and save it in object 'shortBeepButton'
        longBeepButton = QPushButton('long beep', self)
        quitButton = QPushButton('exit', self)
        # adding self in fornt, makes a member variable: cann be accessed from everywhere in the class, even from objects made from the class
        self.label = QLabel('Welcome', self)
        
        shortBeepButton.setGeometry(10, 10, 100, 30)
        longBeepButton.setGeometry(110, 10, 100, 30)
        quitButton.setGeometry(210, 10, 100, 30)
        self.label.setGeometry(10, 40, 500, 70)
        
        # Call back functions: functions that will operate when user pressed a button
        shortBeepButton.clicked.connect(self.shortBeepFunction) # operate def sortBeepFunction
        longBeepButton.clicked.connect(self.longBeepFunction)  # operate def loongBeepFunction
        quitButton.clicked.connect(self.quitFunction)   # operate def quitFunction
        
    def shortBeepFunction(self):
        self.label.setText('Make a beep sound in 1000Hz for 0.5 seconds')   # setText: write text on the label widget
        winsound.Beep(500, 500) # create sound
    
    def longBeepFunction(self):
        self.label.setText('Make a beep sound in 1000Hz for 3 seconds')
        winsound.Beep(500, 3000)
    
    def quitFunction(self):
        self.close()

app = QApplication(sys.argv)    # Create object 'app' to use PyQt
win = BeepSound()   # Create object 'win' using class BeepSound
win.show()  # show window
app.exec_() # make an infinite loop so that program will not shutdown instantly
