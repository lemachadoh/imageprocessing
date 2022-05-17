import cv2 as cv
import numpy as np
import PIL
from matplotlib import pyplot as plt
from picamera import PiCamera
from time import sleep
from PIL import Image, ImageOps

camera = PiCamera()
camera.rotation = 180
camera.capture('/home/silicon-leandra/Desktop/raspcam.jpg')
camera.stop_preview()
rasp_cam = Image.open('/home/silicon-leandra/Desktop/raspcam.jpg')

left = 145 ; top = 0 ; right = 1200 ; bottom = 850

img = rasp_cam.crop((left,top,right,bottom))
img.save('/home/silicon-leandra/Desktop/raspcam.jpg')

planets = cv.imread('/home/silicon-leandra/Desktop/raspcam.jpg')
gray_img = cv.cvtColor(planets, cv.COLOR_BGR2GRAY)

def color(image):
    image =  cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    return image

def gamma_and_threshold(image):
    blur = cv.medianBlur(color(image),7)
    gamma = np.array(255* (blur/255)**2,dtype='uint8')
    threshold = cv.adaptiveThreshold(gamma,177,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV,761,0.5)
    threshold = cv.bitwise_not(threshold)
    return threshold


def dilatation_and_erosion(threshold):
    kernel = np.ones((15,15), np.uint8)
    img_dilation = cv.dilate(gamma_and_threshold(threshold), kernel, iterations=1)
    img_erode = cv.erode(img_dilation,kernel, iterations=1)
    return img_erode

minDist = 150 ; param1 = 100 ; param2 = 30 ; minRadius = 0 ; maxRadius = 0

circles = cv.HoughCircles(
    cv.medianBlur(gray_img, 5),               #8-bit, single-channel, grayscale input image.
    cv.HOUGH_GRADIENT,                        #Detection method
    1,                                        #Inverse ratio of the accumulator resolution to the image resolution
    minDist,                                  #Minimum distance between the centers of the detected circles
    param1=param1,                            #First method-specific parameter
    param2=param2,                            #Second method-specific parameter.
    minRadius=minRadius,
    maxRadius=maxRadius)

if circles is not None:
    n_circulos = 0
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv.circle(
            planets,         #image
            (i[0], i[1]),    #center 
            i[2],            #radius
            (0,255, 0),      #color
            1)               #shift
        
        cv.circle(
            planets,
            (i[0],i[1]),
            3,
            (0,0,255),
            1)
        
        n_circulos = n_circulos + 1
    print('circles found:', n_circulos)

else: 
    print("No circles found")

imagem = dilatation_and_erosion(planets)

ret, labels = cv.connectedComponents(imagem)

label_hue = np.uint8(179 * labels / np.max(labels))
blank_ch = 255 * np.ones_like(label_hue)

labeled_img = cv.merge([label_hue, blank_ch, blank_ch])
labeled_img = cv.cvtColor(labeled_img,cv.COLOR_BGR2GRAY)
labeled_img[label_hue == 0] = 0

''''
cv.imshow("HoughCircles", planets)
cv.waitKey()
cv.destroyAllWindows()

'''

plt.imshow(planets)
plt.title("Circles counted: {0}, objects counted: {1}".format(n_circulos,ret-1))
plt.show()