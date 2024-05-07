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

crp_point = []
#videonun ilk frame ini alıp üzerinde işlem yapacağımız için ilk frame i alıyoruz
frame = cv2.imread("videos/solidWhiteRight.mp4")
def draw_circle(event,x,y,flags,param):
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(frame,(x,y),10,(255,0,0),-1)
        mouseX,mouseY = x,y
        print (mouseX, mouseY)
        crp_point.append([mouseX,mouseY])

def draw_lines(img, points):
    for i in range(len(points)-1):
        cv2.line(img, (points[i][0], points[i][1]), (points[i+1][0], points[i+1][1]), (255, 0, 0), 5)
    cv2.line(img, (points[-1][0], points[-1][1]), (points[0][0], points[0][1]), (255, 0, 0), 5)
    
def area_select(frame):

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_circle)
    while(1):
        cv2.imshow('image', frame)
        # enter a basılırsa çık
        if cv2.waitKey(100) & 0xFF == 13:
            draw_lines(frame,  crp_point)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

    print(crp_point)



def detect_lanes(frame):
    # Convert the frame to grayscale
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Define a region of interest (ROI)
    height, width = edges.shape
    roi_vertices = [crp_point[0], crp_point[1],crp_point[2],crp_point[3] ]
    #crp_pointlere göre kesilmiş resmi göster
    

    mask = np.zeros_like(edges)
    
    #cv2.imshow('mask', mask)

    cv2.fillPoly(mask, np.array([roi_vertices], np.int32), 255)
    masked_edges = cv2.bitwise_and(edges, mask)

  
    
    cv2.imshow('masked_edges', masked_edges)


    #blabla = cv2.bitwise_and(gray,mask)
    #cv2.imshow('blabla', blabla)


    # Apply Hough transform to detect lines
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180 ,100, minLineLength=10, maxLineGap=5)
    
    # Draw the detected lines on the frame
    line_image = np.zeros_like(frame)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0,0), 5)



    # Combine the line image with the original frame
    #result = cv2.bitwise_and(gray,mask)
    result = cv2.addWeighted(frame, 0.8, line_image, 1,0)

    
       #*****************************************************
    IMAGE_H = 223   
    IMAGE_W = 1280

    #src = np.float32([[0,IMAGE_H],[1207, IMAGE_H], [0,0],[IMAGE_W,0]])
    #dst = np.float32([[569,IMAGE_H],[711, IMAGE_H], [0,0],[IMAGE_W,0]])

    src = np.float32([crp_point[0],crp_point[1],crp_point[3], crp_point[2] ])
    dst = np.float32([[300,IMAGE_H],[711, IMAGE_H], [0,0],[IMAGE_W,0]])

    M = cv2.getPerspectiveTransform(src,dst)

    
    
    
    result = cv2.warpPerspective(result, M , (IMAGE_W,IMAGE_H))
    #*****************************************************
    #180 derece çevirme kodu
    result = cv2.rotate(result, cv2.ROTATE_180)

    return result

# Open the video file
#video = cv2.Image("solidWhiteRight.mp4")
#open the image file
video = cv2.VideoCapture("videos/solidWhiteRight.mp4")
#video = cv2.VideoCapture(0)
car_img = cv2.imread("static_car_photos/car_bev1.png")

if car_img is not None:
    # car_img scale %50
    scale_percent = 50 # percent of original size
    width = int(car_img.shape[1]* scale_percent / 100)
    height = int(car_img.shape[0] * scale_percent / 100)
    dim = (width, height)
    car_img = cv2.resize(car_img, dim, interpolation = cv2.INTER_AREA) # 400x400
else:
    print("Failed to load car image.")

# BEYAZ ARKA PLANI SİYAH YAP
car_img[np.where((car_img==[255,255,255]).all(axis=2))] = [0,0,0]
# resim boyutunu değiştirmeden üstten 400 piksel ekle
car_img = np.vstack([np.zeros((400,400,3),dtype=np.uint8),car_img])
# resim boyutunu değiştirmeden sağdan ve soldan 440 piksel ekle
car_img = np.hstack([np.zeros((800,440,3),dtype=np.uint8),car_img,np.zeros((800,440,3),dtype=np.uint8)])

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
    result = cv2.resize(result, (1280, 400))
    
    result = np.vstack([np.zeros((75,1280,3),dtype=np.uint8),result,np.zeros((325,1280,3),dtype=np.uint8)]) 
    #np.zeros((75,1280,3),dtype=np.uint8) şu anlama gelir 75x1280 boyutunda siyah bir resim oluştur
    #np.zeros((325,1280,3),dtype=np.uint8) şu anlama gelir 325x1280 boyutunda siyah bir resim oluştur
    #result = np.vstack([np.zeros((75,1280,3),dtype=np.uint8),result,np.zeros((325,1280,3),dtype=np.uint8)]) #oluşturulan siyah resimleri result resminin üst ve altına ekler
    result = result[0:800, 0:1240] #oluşturulan siyah resimleri result resminin üst ve altına ekler
    result = np.hstack([np.zeros((800,40,3),dtype=np.uint8),result]) #resim boyutunu değiştirmeden sağdan ve soldan 440 piksel ekle
   

    print("result shape:",result.shape)
    #iki resmi birleştir
    result = cv2.addWeighted(result, 0.8, car_img, 1,0)

    # Siyah renk değeri
    color = [0, 0, 0]  # Siyah için RGB değeri

    # Resmin etrafına siyah çerçeve ekleyin


    



    
    cv2.imshow('Original', frame)
    cv2.imshow('Output', result)
   


    # Exit if 'q' is pressed
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

# Release the video file and close all windows
video.release()
cv2.destroyAllWindows()