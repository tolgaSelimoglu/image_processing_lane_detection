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
        if len(crp_point) == 4:
            draw_lines(frame,  crp_point)
        if cv2.waitKey(100) & 0xFF == 13:
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
    lines = cv2.HoughLinesP(masked_edges, 0.1, np.pi/180 ,20, minLineLength=1, maxLineGap=100000)
    
    # Draw the detected lines on the frame
    line_image = np.zeros_like(frame)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            print(x1, y1, x2, y2)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 100,0), 2)



    # Combine the line image with the original frame
    #result = cv2.bitwise_and(gray,mask)



    ################################################33
    result = cv2.addWeighted(frame, 0.8, line_image, 1,0)

    
       #*****************************************************



    #src = np.float32([[0,IMAGE_H],[1207, IMAGE_H], [0,0],[IMAGE_W,0]])
    #dst = np.float32([[569,IMAGE_H],[711, IMAGE_H], [0,0],[IMAGE_W,0]])
    """
    src = np.float32([crp_point[0],crp_point[1],crp_point[3], crp_point[2] ])
    dst = np.float32([[0,0],[0, IMAGE_W], [300,300],[300,600]])
    M = cv2.getPerspectiveTransform(src,dst)
    """
    width_AD = np.sqrt(((crp_point[0][0] - crp_point[1][0]) ** 2) + ((crp_point[0][1] - crp_point[1][1]) ** 2))
    width_BC = np.sqrt(((crp_point[3][0] - crp_point[2][0]) ** 2) + ((crp_point[3][1] - crp_point[2][1]) ** 2))
    maxWidth = max(int(width_AD), int(width_BC))

    height_AB = np.sqrt(((crp_point[0][0] - crp_point[3][0]) ** 2) + ((crp_point[0][1] - crp_point[3][1]) ** 2))
    height_CD = np.sqrt(((crp_point[2][0] - crp_point[1][0]) ** 2) + ((crp_point[2][1] - crp_point[1][1]) ** 2))
    maxHeight = max(int(height_AB), int(height_CD))

    src = np.float32([crp_point[0],crp_point[1],crp_point[3], crp_point[2] ])
    dst = np.float32([[0,0],[maxWidth,0],[0,maxHeight],[maxWidth,maxHeight]])
    
    M = cv2.getPerspectiveTransform(src,dst)
    result = cv2.warpPerspective(result, M , (maxWidth,maxHeight))
    
    
    #line_image = cv2.warpPerspective(line_image, M , (int(maxWidth*0.75),int(maxHeight*0.75)))
    #*****************************************************
    #180 derece çevirme kodu
    #result = cv2.rotate(result, cv2.ROTATE_180)

    #result = cv2.addWeighted(result, 0.8, line_image, 1,0)
    #result = cv2.warpAffine(result, np.float32([[1, 0, -500], [0, 1, -100]]), (result.shape[1], result.shape[0]))
    
    return result

# Open the video file
#video = cv2.Image("solidWhiteRight.mp4")
#open the image file
video = cv2.VideoCapture("videos/solidWhiteRight.mp4")
#video = cv2.VideoCapture(0)

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

   # result = cv2.resize(result, (1280, 400))



    """
    result = np.vstack([np.zeros((75,1280,3),dtype=np.uint8),result,np.zeros((325,1280,3),dtype=np.uint8)]) 
    #np.zeros((75,1280,3),dtype=np.uint8) şu anlama gelir 75x1280 boyutunda siyah bir resim oluştur
    #np.zeros((325,1280,3),dtype=np.uint8) şu anlama gelir 325x1280 boyutunda siyah bir resim oluştur
    #result = np.vstack([np.zeros((75,1280,3),dtype=np.uint8),result,np.zeros((325,1280,3),dtype=np.uint8)]) #oluşturulan siyah resimleri result resminin üst ve altına ekler
    result = result[0:800, 0:1240] #oluşturulan siyah resimleri result resminin üst ve altına ekler
    result = np.hstack([np.zeros((800,40,3),dtype=np.uint8),result]) #resim boyutunu değiştirmeden sağdan ve soldan 440 piksel ekle
   """




    


    
    cv2.imshow('Original', frame)
    cv2.imshow('Output', result)



    # Exit if 'q' is pressed
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

# Release the video file and close all windows
video.release()
cv2.destroyAllWindows()