from imageProcessor import ImageProcessor
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QMessageBox
from PyQt6.QtWidgets import QComboBox, QMainWindow,QFileDialog, QVBoxLayout, QHBoxLayout, QLabel
from PyQt6.QtCore import Qt, QUrl
import sys
import glob
import os

class InitialWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Image Processor V1.1')
        self.setGeometry(300, 300, 300, 200)

        # Create layout
        layout = QVBoxLayout()

        # Add a label for scenario selection
        self.lbl_scenario = QLabel('Select a Video:')
        layout.addWidget(self.lbl_scenario)

        # Add ComboBox for scenarios
        self.cb_scenario = QComboBox(self)
        self.load_video_files()
        layout.addWidget(self.cb_scenario)

        # Add button to run image processor
        self.btn_scenario = QPushButton('Run Image Processor', self)
        self.btn_scenario.clicked.connect(self.runImageProcessor)
        layout.addWidget(self.btn_scenario)

        # Add button to import video
        self.btn_vid = QPushButton('Import Video', self)
        self.btn_vid.clicked.connect(self.openVideoFile)
        layout.addWidget(self.btn_vid)

        # Set the main layout
        self.setLayout(layout)

        self.selected_video = "N/A"

    def load_video_files(self):
        media_folder = './media/vid'
        mp4_files = glob.glob(os.path.join(media_folder, '*.mp4'))

        for file in mp4_files:
            print('FILE_DEBUG' + file)
            self.cb_scenario.addItem(file)

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
        self.lbl_scenario.setText(f"<b style='color: green;'>Video imported. Please run Image Procesor.</b>")

    def runImageProcessor(self):
        if self.selected_video == "N/A":
            self.selected_video = self.cb_scenario.currentText()

        self.close()
        procesor = ImageProcessor(str(self.selected_video))
        procesor.process()
        

    def errorHandler(self):
        QMessageBox.information(self, 'Message', 'Error!')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = InitialWindow()
    window.show()
    sys.exit(app.exec())