from imageProcessor import ImageProcessor
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QMessageBox, QComboBox, QMainWindow,QFileDialog
from PyQt6.QtCore import Qt, QUrl
import sys

class InitialWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Image Processor V1.1')
        self.setGeometry(300, 300, 200, 225) # x, y, width, height

        self.cb_scenario = QComboBox(self)
        self.cb_scenario.addItem('/media/video1.mp4')
        self.cb_scenario.addItem('/media/video2.mp4')
        self.cb_scenario.addItem('/media/video3.mp4')
        self.cb_scenario.addItem('/media/video4.mp4')
        self.cb_scenario.move(60, 25)

        self.btn_scenario = QPushButton('Run Image Processor', self)
        self.btn_scenario.clicked.connect(self.runImageProcessor)
        self.btn_scenario.move(58, 60)

        self.btn_vid = QPushButton('Import Video', self)
        self.btn_vid.clicked.connect(self.openVideoFile)
        self.btn_vid.move(60, 125)

    def openVideoFile(self):
        file_name = QFileDialog.getOpenFileName(
            self,
            "Open File",
            "./",
            "All Files (*);; MP4 Files (*mp4)",
        )
        if file_name:
            self.selected_video = file_name
            self.selected_video = str(self.selected_video).split(',')[0]
            self.selected_video = str(self.selected_video).replace('(', '')
            self.selected_video = str(self.selected_video).replace('\'', '')

        print("DEBUG: Selected File:" + str(self.selected_video))
    
    def runImageProcessor(self):
        procesor = ImageProcessor(str(self.selected_video))
        procesor.process()
        self.close()

    def errorHandler(self):
        QMessageBox.information(self, 'Message', 'Error!')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = InitialWindow()
    window.show()
    sys.exit(app.exec())