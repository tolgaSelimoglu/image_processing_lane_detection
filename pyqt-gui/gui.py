import sys
import subprocess
import cv2
import numpy as np

from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QMessageBox, QComboBox, QMainWindow,QFileDialog
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import Qt, QUrl

class InitialWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Image Processor V1.1')
        self.setGeometry(300, 300, 200, 225) # x, y, width, height

        self.cb_scenario = QComboBox(self)
        self.cb_scenario.addItem('Scenario 1')
        self.cb_scenario.addItem('Scenario 2')
        self.cb_scenario.addItem('Scenario 3')
        self.cb_scenario.addItem('Scenario 4')
        self.cb_scenario.move(60, 25)

        self.btn_scenario = QPushButton('Select Scenario', self)
        self.btn_scenario.clicked.connect(self.runImageProcessor)
        self.btn_scenario.move(58, 60)

        self.btn_vid = QPushButton('Import Video', self)
        self.btn_vid.clicked.connect(self.openVideoFile)
        self.btn_vid.move(60, 125)

        self.btn_img = QPushButton('Import Image', self)
        self.btn_img.clicked.connect(self.openImageFile)
        self.btn_img.move(60, 160)

    def showError(self):
        QMessageBox.information(self, 'Message', 'Error!')

    def openVideoFile(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mkv);;All Files (*)", options=options)
        
        if file_name:
            self.selected_video = file_name

        print("DEBUG: Selected File:" + self.selected_video)
    
    def openImageFile(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image File", "", "Image Files (*.mp4 *.avi *.mkv);;All Files (*)", options=options)
        
        if file_name:
            self.selected_image = file_name

        print("DEBUG: Selected File:" + self.selected_image)

    def runImageProcessor(self):
        self.close()
        self.mainWindow = MainWindow(videoNumber=self.cb_scenario.currentIndex() + 1, videoPath = self.selected_video, imagePath = self.selected_image)
        self.mainWindow.show()

    #def errorHandler(self, code):

class MainWindow(QMainWindow):

    def __init__(self, videoNumber, videoPath, imagePath):
        super().__init__()
        self.videoNumber = videoNumber
        self.videoPath = videoPath
        self.imagePath = imagePath

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

class ImageProcessor:
    def __init__(self, vid_path, img_path):
        print('Image processor object created.')
        self.processing_video_path = path
        self.processing_image_path = img_path

    def area_select(self, image):
        crp_point = [155, 424, 721, 114]
        print("CRP Point :", crp_point)

    def detect_lanes(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert the frame to grayscale
        blurred = cv2.GaussianBlur(gray, (3, 3), 0) # Apply Gaussian blur to reduce noise
        edges = cv2.Canny(blurred, 50, 150) # Apply Canny edge detection

        height, width = edges.shape # Define a region of interest (ROI)
        crp_point = [155, 424, 721, 114]  # Example crp_point values
        roi_vertices = [(crp_point[0], crp_point[1]+crp_point[3]), 
                        (crp_point[0], crp_point[1]),
                        (crp_point[0]+crp_point[2], crp_point[1]),
                        (crp_point[0]+crp_point[2], crp_point[1] + crp_point[3])]

        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, np.array([roi_vertices], np.int32), 255)
        masked_edges = cv2.bitwise_and(edges, mask)
    
        cv2.imshow('masked_edges', masked_edges)
        lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180 ,100, minLineLength=10, maxLineGap=5) # Apply Hough transform to detect lines
        
        line_image = np.zeros_like(frame) # Draw the detected lines on the frame
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

        result = cv2.addWeighted(frame, 0.8, line_image, 1, 0) # Combine the line image with the original frame

        IMAGE_H = 223   
        IMAGE_W = 1280

        src = np.float32([[0, IMAGE_H], [1207, IMAGE_H], [0, 0], [IMAGE_W, 0]])
        dst = np.float32([[569, IMAGE_H], [711, IMAGE_H], [0, 0], [IMAGE_W, 0]])

        M = cv2.getPerspectiveTransform(src, dst)

        result = result[450:(450+IMAGE_H), 0:IMAGE_W]

        result = cv2.warpPerspective(result, M, (IMAGE_W, IMAGE_H))
        
        return result

    def image_process(self):
        crp_point = [0, 0, 0, 0]

        video = cv2.VideoCapture(self.processing_video_path)
        car_img = cv2.imread(self.processing_image_path)

        # car_img scale %50
        scale_percent = 50 # percent of original size
        width = int(car_img.shape[1] * scale_percent / 100)
        height = int(car_img.shape[0] * scale_percent / 100)
        dim = (width, height)
        car_img = cv2.resize(car_img, dim, interpolation=cv2.INTER_AREA) # 400x400
        car_img[np.where((car_img==[255,255,255]).all(axis=2))] = [0,0,0] # Beyaz arka plani siyah yap
        car_img = np.vstack([np.zeros((400,400,3), dtype=np.uint8), car_img]) # resim boyutun degistirmeden ustten 400 piksel ekle
        car_img = np.hstack([np.zeros((800,440,3), dtype=np.uint8), car_img, np.zeros((800,440,3), dtype=np.uint8)]) # sagdan soldan 440 piksel ekle

        while True:
            ret, frame = video.read() # Read a frame from the video

            if not ret:
                break
            self.area_select(frame)
            break
        while True:
            ret, frame = video.read() # Process the frame

            if not ret:
                break
            result = self.detect_lanes(frame)
            result = cv2.resize(result, (1280, 400))
            
            result = np.vstack([np.zeros((75,1280,3), dtype=np.uint8), result, np.zeros((325,1280,3), dtype=np.uint8)]) 
            result = result[0:800, 0:1240]
            result = np.hstack([np.zeros((800,40,3), dtype=np.uint8), result]) 
        

            print("result shape:", result.shape)
            result = cv2.addWeighted(result, 0.8, car_img, 1,0)

            cv2.imshow('Original', frame)
            cv2.imshow('Output', result)

            if cv2.waitKey(100) & 0xFF == ord('q'):
                break # Exit if 'q' is pressed

        # Release the video file and close all windows
        video.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = InitialWindow()
    window.show()
    sys.exit(app.exec_())