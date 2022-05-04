from email.mime import image
from imaplib import Time2Internaldate
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
#from picamera import PiCamera
from time import sleep
from PIL import Image

camera = PiCamera()

rasp_cam = Image.open('/home/silicon/Desktop/raspcam.jpg')

def take_and_crop_picture():

    global output
    output = ("/home/silicon/Desktop/raspcam.jpg")
    camera.rotation = 180
    camera.capture(output)
    camera.stop_preview()

    image_output = Image.open(output)

    image = image_output.crop(
        left = 230,
        top = 170, 
        right = 705, 
        bottom = 700)

    image.save()

    image = cv.imread('')

    return image

img = cv.imread(r'C:\Users\lwmachado\Desktop\fotos\moedas.jpg')



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

def circle(image):

    minDist = 35; param1 = 600; param2 = 100 ;minRadius = 0 ;maxRadius = 0

    cv.HoughCircles(
        color(image),
        cv.HOUGH_GRADIENT, 
        1, 
        minDist, 
        param1=param1, 
        param2=param2, 
        minRadius=minRadius, 
        maxRadius=maxRadius
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))

        for i in circles[0,:]:
                cv.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
        else: 
            print("No circles found")
    return circles

imagem = dilatation_and_erosion(img)

ret, labels = cv.connectedComponents(imagem)

label_hue = np.uint8(179 * labels / np.max(labels))
blank_ch = 255 * np.ones_like(label_hue)

labeled_img = cv.merge([label_hue, blank_ch, blank_ch])
labeled_img = cv.cvtColor(labeled_img,cv.COLOR_BGR2GRAY)
labeled_img[label_hue == 0] = 0


'''
shape_circle = circles.shape[1:2:]


#plt.imshow(labeled_img)
plt.imshow(img)
plt.legend()
plt.title("Circles counted: ".format(shape_circle))
plt.show()
print('objects number is:', ret-1)
print(circles.shape[1:2])

'''
print(type(img))