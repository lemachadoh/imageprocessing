from collections import Counter
from sklearn.cluster import KMeans
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from PIL import Image, ImageOps
from picamera import PiCamera

camera = PiCamera()
camera.rotation = 240
camera.capture('/home/silicon-leandra/Desktop/raspcam.jpg')
camera.stop_preview()
rasp_cam = Image.open('/home/silicon-leandra/Desktop/raspcam.jpg')

left = 145 ; top = 0 ; right = 1200 ; bottom = 850

img = rasp_cam.crop((left,top,right,bottom))
img.save('/home/silicon-leandra/Desktop/raspcam.jpg')

planets = cv.imread('/home/silicon-leandra/Desktop/raspcam.jpg')

hsv = cv.cvtColor(planets, cv.COLOR_BGR2HSV)


#green mask
green_low = np.array([36,25,25], np.uint8)
green_up = np.array([70,255,255], np.uint8)
green_mask = cv.inRange(hsv, green_low, green_up)

#blue mask
blue_low = np.array([94,80,2], np.uint8)
blue_up = np.array([120,255,255], np.uint8)
blue_mask = cv.inRange(hsv, blue_low, blue_up)

#red mask
red_low = np.array([136,87,111], np.uint8)
red_up = np.array([180,255,255], np.uint8)
red_mask = cv.inRange(hsv, red_low, red_up)
    
kernal = np.ones((5,5), 'uint8')
    
green_mask = cv.dilate(green_mask, kernal)
res_green = cv.bitwise_and(planets, planets,
                            mask = green_mask)


blue_mask = cv.dilate(blue_mask, kernal)
res_blue = cv.bitwise_and(planets, planets,
                            mask = blue_mask)


red_mask = cv.dilate(red_mask, kernal)
res_red = cv.bitwise_and(planets, planets,
                            mask = red_mask)

#green
contours, hierarchy = cv.findContours(green_mask,
                                        cv.RETR_TREE,
                                        cv.CHAIN_APPROX_SIMPLE)


for pic, contour in enumerate(contours):
    area = cv.contourArea(contour)
    if(area>300):
        x,y,w,h = cv.boundingRect(contour)
        planets = cv.rectangle(planets, (x,y),
                                 (x + w, y + h),
                                 (0,255,0), 2)
        cv.putText(planets, "green", (x,y),
                       cv.FONT_HERSHEY_SIMPLEX,
                       1.0, (0,255,0))
      
#blue

contours, hierarchy = cv.findContours(blue_mask,
                                        cv.RETR_TREE,
                                        cv.CHAIN_APPROX_SIMPLE)


for pic, contour in enumerate(contours):
    area = cv.contourArea(contour)
    if(area>300):
        x,y,w,h = cv.boundingRect(contour)
        planets = cv.rectangle(planets, (x,y),
                                 (x + w, y + h),
                                 (255,0,0), 2)
        cv.putText(planets, "blue", (x,y),
                       cv.FONT_HERSHEY_SIMPLEX,
                       1.0, (255,0,0))
        

#red
contours, hierarchy = cv.findContours(red_mask,
                                        cv.RETR_TREE,
                                        cv.CHAIN_APPROX_SIMPLE)

for pic, contour in enumerate(contours):
    area = cv.contourArea(contour)
    if(area>300):
        x,y,w,h = cv.boundingRect(contour)
        planets = cv.rectangle(planets, (x,y),
                                 (x + w, y + h),
                                 (0,0,255), 2)
        cv.putText(planets, "red", (x,y),
                       cv.FONT_HERSHEY_SIMPLEX,
                       1.0, (0,0,255))
        



plt.imshow(planets)
plt.show()
    
    





















