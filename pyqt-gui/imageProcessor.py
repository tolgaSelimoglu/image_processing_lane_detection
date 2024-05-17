import numpy as np
import cv2

class ImageProcessor:
    def __init__(self, videoPath):
        self.videoPath = videoPath

    def draw_circle(self, event, x, y, flags, param):
        global mouseX, mouseY

        if event == cv2.EVENT_LBUTTONDBLCLK:
            cv2.circle(self.frame, (x, y), 10, (255, 0, 0), -1)
            mouseX, mouseY = x, y
            print(mouseX, mouseY)
            self.crp_point.append([mouseX, mouseY])

    def draw_lines(self, img, points):
        for i in range(len(points) - 1):
            cv2.line(img, (points[i][0], points[i][1]), (points[i + 1][0], points[i + 1][1]), (255, 0, 0), 5)
        cv2.line(img, (points[-1][0], points[-1][1]), (points[0][0], points[0][1]), (255, 0, 0), 5)

    def area_select(self):
        cv2.namedWindow('image', cv2.WND_PROP_FULLSCREEN)
        cv2.setMouseCallback('image', self.draw_circle)

        #screen_width = 1366  # Default width, you may need to update this
        #screen_height = 768  # Default height, you may need to update this

        cv2.setWindowProperty("image",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
       # cv2.moveWindow('image', int(screen_width * 0.25), int(screen_height * 0.25))  # Position the window at (10% of screen width, 10% of screen height)

        while(1):
            cv2.imshow('image', self.frame)

            if len(self.crp_point) == 4:
                self.draw_lines(self.frame, self.crp_point)

            if cv2.waitKey(100) & 0xFF == 13:  # kill process if enter pressed
                break

        cv2.destroyAllWindows()

        print(self.crp_point)

    def detect_lanes(self):
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)  # Apply Gaussian blur to reduce noise
        edges = cv2.Canny(blurred, 50, 150)  # Apply Canny edge detection
        height, width = edges.shape  # Define a region of interest (ROI)
        roi_vertices = [self.crp_point[0], self.crp_point[1], self.crp_point[2], self.crp_point[3]]

        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, np.array([roi_vertices], np.int32), 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        cv2.namedWindow('masked_edges')
        cv2.imshow('masked_edges', masked_edges)

        screen_width = 1920  # Default width, you may need to update this
        screen_height = 1080  # Default height, you may need to update this

        cv2.moveWindow('masked_edges', int(screen_width * 0.5), int(screen_height * 0.1))  # Position the window at (50% of screen width, 10% of screen height)

        lines = cv2.HoughLinesP(masked_edges, 0.1, np.pi / 180, 20, minLineLength=1, maxLineGap=100000)  # Apply Hough transform to detect lines

        line_image = np.zeros_like(self.frame)  # Draw the detected lines on the frame

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                print(x1, y1, x2, y2)
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 100, 0), 2)

        result = cv2.addWeighted(self.frame, 0.8, line_image, 1, 0)

        width_AD = np.sqrt(((self.crp_point[0][0] - self.crp_point[1][0]) ** 2) + ((self.crp_point[0][1] - self.crp_point[1][1]) ** 2))
        width_BC = np.sqrt(((self.crp_point[3][0] - self.crp_point[2][0]) ** 2) + ((self.crp_point[3][1] - self.crp_point[2][1]) ** 2))
        maxWidth = max(int(width_AD), int(width_BC))

        height_AB = np.sqrt(((self.crp_point[0][0] - self.crp_point[3][0]) ** 2) + ((self.crp_point[0][1] - self.crp_point[3][1]) ** 2))
        height_CD = np.sqrt(((self.crp_point[2][0] - self.crp_point[1][0]) ** 2) + ((self.crp_point[2][1] - self.crp_point[1][1]) ** 2))
        maxHeight = max(int(height_AB), int(height_CD))

        src = np.float32([self.crp_point[0], self.crp_point[1], self.crp_point[3], self.crp_point[2]])
        dst = np.float32([[0, 0], [maxWidth, 0], [0, maxHeight], [maxWidth, maxHeight]])

        M = cv2.getPerspectiveTransform(src, dst)
        result = cv2.warpPerspective(result, M, (maxWidth, maxHeight))

        return result

    def process(self):
        self.crp_point = []
        self.frame = cv2.imread(self.videoPath)  # read first frame
        video = cv2.VideoCapture(self.videoPath)
        # video = cv2.VideoCapture(0)

        while True:
            ret, self.frame = video.read()  # Read a frame from the video

            if not ret:
                break

            self.area_select()
            break

        while True:
            ret, self.frame = video.read()  # Process the frame

            if not ret:
                break

            result = self.detect_lanes()

            cv2.namedWindow('Original')
            cv2.imshow('Original', self.frame)
            screen_width = 1920  # Default width, you may need to update this
            screen_height = 1080  # Default height, you may need to update this
            cv2.moveWindow('Original', int(screen_width * 0.1), int(screen_height * 0.6))  # Position the window at (10% of screen width, 60% of screen height)

            cv2.namedWindow('Output')
            cv2.imshow('Output', result)
            cv2.moveWindow('Output', int(screen_width * 0.5), int(screen_height * 0.6))  # Position the window at (50% of screen width, 60% of screen height)

            if cv2.waitKey(100) & 0xFF == ord('q'):
                break  # Exit if 'q' is pressed

        video.release()
        cv2.destroyAllWindows()  # Release the video file and close all windows
