import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = cv.imread(r"C:\Users\lwmachado\Desktop\fotos\moedas.jpg")

gray = cv.cvtColor(image, cv2.COLOR_BGR2GRAY)

minDist = 50 ; param1 = 600 ; param2 = 100 ; minRadius = 0 ; maxRadius = 0

circles = cv.HoughCircles(
    color(img),        #8-bit, single-channel, grayscale input image.
    cv.HOUGH_GRADIENT, #Detection method
    2,                 #Inverse ratio of the accumulator resolution to the image resolution
    minDist,           #Minimum distance between the centers of the detected circles
    param1=param1,     #First method-specific parameter
    param2=param2,     #Second method-specific parameter.
    minRadius=minRadius,
    maxRadius=maxRadius)

n_circulos = 0

if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv.circle(
            image,           #image
            (i[0], i[1]),    #center 
            i[2],            #radius
            (0,255, 255),    #color
            2)               #shift
        n_circulos = n_circulos + 1
else: 
    print("No circles found")


