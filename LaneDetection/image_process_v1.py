import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
try:
    import cv2
except ImportError:
    install("opencv-python")
try:
    import numpy as np
except ImportError:
    install("numpy")
try:
    import matplotlib.pyplot as plt
except ImportError:
    install("matplotlib")


crp_point = [0,0,0,0]

def area_select(image):
   
    #r = cv2.selectROI("Select area", image)
    
    #crp_point[0],crp_point[1],crp_point[2],crp_point[3] = int(r[0]),int(r[1]),  int(r[2]),int(r[3])
    crp_point[0],crp_point[1],crp_point[2],crp_point[3] = 155,424,721,114
    print("CRP Point :",crp_point)

def detect_lanes(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Define a region of interest (ROI)
    height, width = edges.shape
    roi_vertices = [(crp_point[0], crp_point[1]+crp_point[3]), (crp_point[0], crp_point[1]),(crp_point[0]+crp_point[2], crp_point[1]),(crp_point[0]+crp_point[2], crp_point[1]+crp_point[3]) ]
    mask = np.zeros_like(edges)
    
    #cv2.imshow('mask', mask)

    cv2.fillPoly(mask, np.array([roi_vertices], np.int32), 255)
    masked_edges = cv2.bitwise_and(edges, mask)
 
    cv2.imshow('masked_edges', masked_edges)
    # Apply Hough transform to detect lines
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180 ,100, minLineLength=10, maxLineGap=5)
    
    # Draw the detected lines on the frame
    line_image = np.zeros_like(frame)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0,0), 5)



    # Combine the line image with the original frame
    result = cv2.addWeighted(frame, 0.8, line_image, 1,0)

       #*****************************************************
    IMAGE_H = 223   
    IMAGE_W = 1280

    src = np.float32([[0,IMAGE_H],[1207, IMAGE_H], [0,0],[IMAGE_W,0]])
    dst = np.float32([[569,IMAGE_H],[711, IMAGE_H], [0,0],[IMAGE_W,0]])

    M = cv2.getPerspectiveTransform(src,dst)

    result = result[450:(450+IMAGE_H), 0:IMAGE_W]

    result = cv2.warpPerspective(result, M , (IMAGE_W,IMAGE_H))
    #*****************************************************

    return result

# Open the video file
#video = cv2.Image("solidWhiteRight.mp4")
#open the image file
video = cv2.VideoCapture("videos/solidWhiteRight.mp4")
#video = cv2.VideoCapture(0)
car_img = cv2.imread("static_car_photos/car_bev1.png")

video_frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
scale_percent = 67  # Örneğin, orijinal boyutun %50'si
width = int(car_img.shape[1] * scale_percent / 100)
height = int(car_img.shape[0] * scale_percent / 100)
dim = (width, height)
car_img = cv2.resize(car_img, dim, interpolation=cv2.INTER_AREA)
while True:
    # Read a frame from the video
    ret, frame = video.read()

    if not ret:
        break
    area_select(frame)
    break
while True:
    # Process the frame
    ret, frame = video.read()

    if not ret:
        break
    result = detect_lanes(frame)
    top, bottom, left, right = [0, 500, 100, 0]  # Örneğin, her bir kenar için 50 piksel

    # Siyah renk değeri
    color = [0, 0, 0]  # Siyah için RGB değeri

    # Resmin etrafına siyah çerçeve ekleyin
    result = cv2.copyMakeBorder(result, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    result[188:488,580:680] = car_img[20:320,391:491]

    # Apply Canny edge detection
   # edges = cv2.Canny(blurred, 50, 150)

    cv2.imshow('Lane Detection', result)
    cv2.imshow('Original', frame)

    
    #cv2.imshow('gray', gray)
   # cv2.imshow('blurred', blurred)
    #cv2.imshow('edges', edges)
    

 

    # Exit if 'q' is pressed
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

# Release the video file and close all windows
video.release()
cv2.destroyAllWindows()