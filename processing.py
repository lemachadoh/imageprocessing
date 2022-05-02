import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from picamera import PiCamera
from time import sleep
from PIL import Image


camera = PiCamera()
camera.capture('/home/silicon/Desktop/raspcam.jpg')
rasp_cam = Image.open('/home/silicon/Desktop/raspcam.jpg')


left = 230
top = 170
right = 705
bottom = 700

img_crop = rasp_cam.crop((left,top,right,bottom))
img_crop.save('/home/silicon/Desktop/raspcam.jpg')


img = cv.imread('/home/silicon/Desktop/raspcam.jpg')

def color(image):
    image =  cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    return image

def gamma_and_threshold(image):
    blur = cv.medianBlur(color(image),7)
    gamma = np.array(255* (blur/255)**0.15,dtype='uint8')
    threshold = cv.adaptiveThreshold(gamma,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,601,10)
    threshold = cv.bitwise_not(threshold)
    return threshold


def dilatation_and_erosion(threshold):
    kernel = np.ones((15,15), np.uint8)
    img_dilation = cv.dilate(gamma_and_threshold(threshold), kernel, iterations=1)
    img_erode = cv.erode(img_dilation,kernel, iterations=1)
    return img_erode



img = dilatation_and_erosion(img)


ret, labels = cv.connectedComponents(img)

label_hue = np.uint8(179 * labels / np.max(labels))
blank_ch = 255 * np.ones_like(label_hue)

labeled_img = cv.merge([label_hue, blank_ch, blank_ch])
labeled_img = cv.cvtColor(labeled_img,cv.COLOR_BGR2GRAY)
labeled_img[label_hue == 0] = 0

plt.imshow(labeled_img)
plt.show()
print('objects number is:', ret-1)

