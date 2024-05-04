import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QMessageBox, QComboBox, QMainWindow
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import Qt, QUrl

class InitialWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Image Processor V1.1')
        self.setGeometry(300, 300, 200, 275)

        self.btn_scenario = QPushButton('Select Scenario', self)
        self.btn_scenario.clicked.connect(self.openNewWindow)
        self.btn_scenario.move(60, 200)

        self.cb_scenario = QComboBox(self)
        self.cb_scenario.addItem('Scenario 1')
        self.cb_scenario.addItem('Scenario 2')
        self.cb_scenario.addItem('Scenario 3')
        self.cb_scenario.addItem('Scenario 4')
        self.cb_scenario.move(60, 100)

    def showError(self):
        QMessageBox.information(self, 'Message', 'Error!')

    def openNewWindow(self):
        self.close()
        
        self.mainWindow = MainWindow(videoNumber=self.cb_scenario.currentIndex() + 1)
        self.mainWindow.show()

class MainWindow(QMainWindow):

    def __init__(self, videoNumber):
        super().__init__()
        self.videoNumber = videoNumber
        print('DEBUG: ' + str(videoNumber))
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Image Processor V1.1')
        self.setGeometry(300, 300, 800, 600)

        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.videoWidget = QVideoWidget()

        self.setCentralWidget(self.videoWidget)

        video_path = "/media/lenovo/Depo/Github/pyqt-goruntuisleme/pyqt-gui/media/scenario_" + str(self.videoNumber) + ".mp4"  
        print("DEBUG: " + video_path)   

        self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(video_path)))
        self.mediaPlayer.setVideoOutput(self.videoWidget)
        self.mediaPlayer.play()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = InitialWindow()
    window.show()
    sys.exit(app.exec_())
