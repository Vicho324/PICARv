import cv2 as cv
import numpy as np
'''
def make_coordinates()
def average_slope_intercept(image,lines):
    left_fit = []
    right_fit = []
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
'''
def canny(image):
    gray = cv.cvtColor(image,cv.COLOR_RGB2GRAY)
    blur =cv.GaussianBlur(gray,(5,5),0)
    canny =cv.Canny(blur,50,150)
    return canny
def region_of_interest(image):
    height = image.shape[0]
    polygon = np.array([[(0,height),(700,height),(300,250)]])
    mask = np.zeros_like(image)
    cv.fillPoly(mask,polygon,255)
    masked_image = cv.bitwise_and(image,mask)
    return masked_image
def displayLines(image,lines):
    line_image= np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            cv.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
    return line_image



cam = cv.VideoCapture(0)


while True:
    _,frame = cam.read()
    lane_image = np.copy(frame)
    canny_image = canny(lane_image)
    cropped_image = region_of_interest(canny_image)
    lines = cv.HoughLinesP(cropped_image,2,np.pi/180, 100,np.array([]),minLineLength=40, maxLineGap=5)
    #averaged_lines = average_slope_intercept(lane_image,line)
    line_image = displayLines(lane_image,lines)
    combo_image = cv.addWeighted(lane_image, 0.8, line_image, 1,1)

    cv.imshow('result3',combo_image)
    cv.imshow("result2",line_image)
    cv.imshow("result",cropped_image)


    key = cv.waitKey(1)
    if key == ord('q'):
        break


cam.release()
cv.destroyAllWindows()