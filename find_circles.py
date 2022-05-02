import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread(r"C:\Users\lwmachado\Desktop\fotos\moedas.jpg")
output = image.copy()
height, width = image.shape[:2]
maxWidth = int(width/10)
minWidth = int(width/20)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

minDist = 35
param1 = 600
param2 = 100
minRadius = 0
maxRadius = 0

circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)


if circles is not None:

    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)


cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()


