from email.mime import image
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from time import sleep
from PIL import Image

planets = cv.imread(r'C:\Users\lwmachado\Desktop\fotos\formas_geometricas.jpg')


def hsv(image):

    def nothing(x):
        pass


    cv.namedWindow('image')

# Create trackbars for color change
# Hue is from 0-179 for Opencv
    cv.createTrackbar('HMin', 'image', 0, 179, nothing)
    cv.createTrackbar('SMin', 'image', 0, 255, nothing)
    cv.createTrackbar('VMin', 'image', 0, 255, nothing)
    cv.createTrackbar('HMax', 'image', 0, 179, nothing)
    cv.createTrackbar('SMax', 'image', 0, 255, nothing)
    cv.createTrackbar('VMax', 'image', 0, 255, nothing)

# Set default value for Max HSV trackbars
    cv.setTrackbarPos('HMax', 'image', 179)
    cv.setTrackbarPos('SMax', 'image', 255)
    cv.setTrackbarPos('VMax', 'image', 255)

# Initialize HSV min/max values
    hMin = sMin = vMin = hMax = sMax = vMax = 0
    phMin = psMin = pvMin = phMax = psMax = pvMax = 0

    while(1):
    # Get current positions of all trackbars
        hMin = cv.getTrackbarPos('HMin', 'image')
        sMin = cv.getTrackbarPos('SMin', 'image')
        vMin = cv.getTrackbarPos('VMin', 'image')
        hMax = cv.getTrackbarPos('HMax', 'image')
        sMax = cv.getTrackbarPos('SMax', 'image')
        vMax = cv.getTrackbarPos('VMax', 'image')

        lower_h = open('lower_h.txt','w')
        lower_s = open('lower_s.txt','w')
        lower_v = open('lower_v.txt','w')
        upper_h = open('upper_h.txt','w')
        upper_s = open('upper_s.txt','w')
        upper_v = open('upper_v.txt','w')


    # Set minimum and maximum HSV values to display
        lower = np.array([hMin, sMin, vMin])
        upper = np.array([hMax, sMax, vMax])

        lower_h.write(str(lower[0])); lower_s.write(str(lower[1]));lower_v.write(str(lower[2]))
        upper_h.write(str(upper[0])); upper_s.write(str(upper[1]));upper_v.write(str(upper[2]))

    # Convert to HSV format and color threshold
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, lower, upper)
        result = cv.bitwise_and(image, image, mask=mask)

    # Print if there is a change in HSV value
        if((phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax)):
       
            phMin = hMin
            psMin = sMin
            pvMin = vMin
            phMax = hMax
            psMax = sMax
            pvMax = vMax

    # Display result image
        cv.imshow('image', result)
        key = cv.waitKey(10) & 0xFF

        if key == ord('s'):
            cv.imwrite('pic2.jpg', result)
            break

    return result

hsv(planets)

# green mask
# transformar o arquivo str em array e colocar nas variaveis
pic2 = cv.imread(r'C:\Users\lwmachado\pic2.jpg')

with open('lower_h.txt','r') as hm:
    low_h = hm.read()
    hm.close()
with open('lower_s.txt','r') as sm:
    low_s = sm.read()
    sm.close()
with open('lower_v.txt','r') as vm:
    low_v = vm.read()
    vm.close()
lower = np.array([low_h, low_s, low_v],np.uint8)

######################################################
with open('upper_h.txt','r') as h:
    upper_h = h.read()
    h.close()
with open('upper_s.txt','r') as s:
    upper_s = s.read()
    s.close()
with open('upper_v.txt','r') as s:
    upper_v = s.read()
    s.close()

upper = np.array([upper_h, upper_s, upper_v],np.uint8)


green_mask = cv.inRange(pic2, lower, upper)
res_green = cv.bitwise_and(pic2, pic2,
                            mask = green_mask)

# green
contours, hierarchy = cv.findContours(green_mask,
                                      cv.RETR_TREE,
                                      cv.CHAIN_APPROX_SIMPLE)


for pic, contour in enumerate(contours):
    area = cv.contourArea(contour)
    if(area > 300):
        x, y, w, h = cv.boundingRect(contour)
        pic2 = cv.rectangle(pic2, (x, y),
                               (x + w, y + h),
                               (0, 255, 0), 2)
        cv.putText(pic2, "green", (x, y),
                   cv.FONT_HERSHEY_SIMPLEX,
                   1.0, (0, 255, 0))

cv.imshow("colors", pic2)
cv.waitKey(0) 
cv.destroyAllWindows()
