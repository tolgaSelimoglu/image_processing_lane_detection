import cv2
global photo_path
global crp_points
crp_points = []


def draw_circle(event,x,y):
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img,(x,y),30,(255,0,0),-1)
        mouseX,mouseY = x,y
        print (mouseX, mouseY)
        crp_points.append([mouseX,mouseY])

def draw_lines(img, points):
    for i in range(len(points)-1):
        cv2.line(img, (points[i][0], points[i][1]), (points[i+1][0], points[i+1][1]), (255, 0, 0), 5)
    cv2.line(img, (points[-1][0], points[-1][1]), (points[0][0], points[0][1]), (255, 0, 0), 5)
    
def area_select(photo_path):
    global img
    img = cv2.imread(photo_path)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_circle)
    while(1):
        cv2.imshow('image', img)
        # enter a basılırsa çık
        if cv2.waitKey(100) & 0xFF == 13:
            draw_lines(img,  crp_points)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

    print(crp_points)
    
       

