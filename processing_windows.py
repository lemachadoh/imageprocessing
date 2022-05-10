#codigo teste usado no windows 
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
#from picamera import PiCamera
from time import sleep
from PIL import Image

'''''
camera = PiCamera()
camera.start_preview()
sleep(1)
camera.rotation = 180
camera.capture('/home/silicon/Desktop/raspcam.jpg')
camera.stop_preview()

rasp_cam = Image.open('/home/silicon/Desktop/raspcam.jpg')

left = 230  ; top = 170 ; right = 705 ; bottom = 700

img_crop = rasp_cam.crop((left,top,right,bottom))
img_crop.save('/home/silicon/Desktop/raspcam.jpg')
'''''
img = cv.imread(r'C:\Users\lwmachado\Desktop\fotos\formas.png')

def color(image):
    image =  cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    return image

def gamma_and_threshold(image):
    blur = cv.medianBlur(color(image),7)
    gamma = np.array(255* (blur/255)**0.15,dtype='uint8')
    threshold = cv.adaptiveThreshold(gamma,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,201,1)
    threshold = cv.bitwise_not(threshold)
    return threshold


def dilatation_and_erosion(threshold):
    kernel = np.ones((15,15), np.uint8)
    img_dilation = cv.dilate(gamma_and_threshold(threshold), kernel, iterations=1)
    img_erode = cv.erode(img_dilation,kernel, iterations=1)
    return img_erode

minDist = 60 ; param1 = 600 ; param2 = 100 ; minRadius = 0 ; maxRadius = 0

circles = cv.HoughCircles(
    img,                    #8-bit, single-channel, grayscale input image.
    cv.HOUGH_GRADIENT,      #Detection method
    2,                      #Inverse ratio of the accumulator resolution to the image resolution
    minDist,                #Minimum distance between the centers of the detected circles
    param1=param1,          #First method-specific parameter
    param2=param2,          #Second method-specific parameter.
    minRadius=minRadius, 
    maxRadius=maxRadius)

circles = np.uint16(np.around(circles))

n_circulos = 0

if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv.circle(
            img,             #image
            (i[0], i[1]),    #center 
            i[2],            #radius
            (0,255, 255),    #color
            2)               #shift
    n_circulos = n_circulos + 1
else: 
    print("No circles found")
    
imagem = dilatation_and_erosion(img)

ret, labels = cv.connectedComponents(imagem)

label_hue = np.uint8(179 * labels / np.max(labels))
blank_ch = 255 * np.ones_like(label_hue)

labeled_img = cv.merge([label_hue, blank_ch, blank_ch])
labeled_img = cv.cvtColor(labeled_img,cv.COLOR_BGR2GRAY)
labeled_img[label_hue == 0] = 0


#plt.imshow(labeled_img)
plt.imshow(img)
plt.legend()
plt.title("Circles counted: {0} ".format(n_circulos))
plt.show()
print('objects number is:', ret-1)


