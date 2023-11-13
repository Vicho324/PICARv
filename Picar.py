import cv2 as cv
import numpy as np
import math

offset = 1

def rectify(offset):


    if offset == 0:
        angle = 70
    else:
        angle = (90-int(math.degrees(math.atan(7/offset))))*-1+70
        
    return angle





rectify(1)


def make_coordinates(image,line_parameters):
    slope,intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    print(x1,x2)
    return np.array([x1,y1,x2,y2])
    


def average_slope_intercept(image,lines):
    left_fit = []
    right_fit = []
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            parameters = np.polyfit((x1,x2),(y1,y2),1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope < 0:
                left_fit.append((slope,intercept))
            else:
                right_fit.append((slope,intercept))
    left_fit_average =np.average(left_fit,axis = 0)
    right_fit_average =np.average(right_fit,axis = 0)
    left_line = make_coordinates(image,left_fit_average)
    right_line = make_coordinates(image,right_fit_average)
    return np.array([left_line,right_line])


def canny(image):
    gray = cv.cvtColor(image,cv.COLOR_RGB2GRAY)
    blur =cv.GaussianBlur(gray,(5,5),0)
    canny =cv.Canny(blur,50,150)
    return canny
def display_lines(image,lines):
    line_image= np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            cv.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
            print(line)
    return line_image
def region_of_interest(image):
    height = image.shape[0]
    polygon = np.array([[(0,height),(1000,height),(250,250)]])
    mask = np.zeros_like(image)
    cv.fillPoly(mask,polygon,255)
    masked_image = cv.bitwise_and(image,mask)
    return masked_image

def range_lines(image,lines):
    average = 0
    line_image= np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            print(line)
            average = x2-x1

    return average



cam = cv.VideoCapture(0)


while True:
    _,frame = cam.read()
    lane_image = np.copy(frame)
    canny_image = canny(lane_image)
    cropped_image = region_of_interest(canny_image)
    lines = cv.HoughLinesP(cropped_image,2,np.pi/180, 100,np.array([]),minLineLength=40, maxLineGap=5)
    #averaged_lines = average_slope_intercept(lane_image,lines)
    line_image = display_lines(lane_image,lines)
    combo_image = cv.addWeighted(lane_image, 0.8, line_image, 1,1)
    if range_lines(lane_image,lines) != 0:
        print(range_lines(lane_image,lines))

    bw.speed = 40
    fw.turn(rectify(offset))

    cv.imshow('result3',combo_image)
    cv.imshow("result2",line_image)
    cv.imshow("result",cropped_image)


    key = cv.waitKey(1)
    if key == ord('q'):
        break


cam.release()
cv.destroyAllWindows()

