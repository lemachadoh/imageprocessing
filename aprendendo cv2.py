#mportando bibliotecas
from tracemalloc import stop
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

#importando a imagem

img = cv.imread(r"C:\Users\lwmachado\Desktop\teste3.jpeg")
#defiicão das variaveis img em rgb e grayscale
img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
img_rgb = cv.cvtColor(img,cv.COLOR_BGR2RGB)

#plot imagem rgb
plt.subplot(1,1,1)
plt.imshow(img_rgb)
#plt.show()

#convertendo em grayscale
plt.subplot(1,1,1)
plt.title('grayscale img')
plt.imshow(img_gray, cmap = 'gray', vmin=0,vmax=255)
#plt.show()

#ajustando contraste, y=1.2

contraste_gamma = np.array(255* (img_gray/255)**0.4,dtype='uint8')
plt.subplot(1,1,1)
plt.title('correção gamma y=1.2')
plt.imshow(contraste_gamma,cmap='gray',vmin=0,vmax=255)
#plt.show()

#histograma
contraste_hist = cv.equalizeHist(img_gray)
plt.subplot(1,1,1)
plt.title("correção histograma")
plt.imshow(contraste_hist,cmap='gray', vmin=0,vmax=255)
#plt.show()

thresh = cv.adaptiveThreshold(contraste_gamma, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 601, 23)
thresh = cv.bitwise_not(thresh)
plt.subplot(1,1,1),
plt.title('Threshold')
plt.imshow(thresh, cmap="gray", vmin=0, vmax=255)
plt.show()


# Dilatation et erosion
kernel = np.ones((15,15), np.uint8)
img_dilation = cv.dilate(thresh, kernel, iterations=1)
img_erode = cv.erode(img_dilation,kernel, iterations=1)

# clean all noise after dilatation and erosion
img_erode = cv.medianBlur(img_erode, 7)
plt.subplot(1,1,1)
plt.title('Dilatation + erosion')
plt.imshow(img_erode, cmap="gray", vmin=0, vmax=255)
plt.show()


# Labeling

ret, labels = cv.connectedComponents(img_erode)
label_hue = np.uint8(179 * labels / np.max(labels))
blank_ch = 255 * np.ones_like(label_hue)
labeled_img = cv.merge([label_hue, blank_ch, blank_ch])
labeled_img = cv.cvtColor(labeled_img, cv.COLOR_HSV2BGR)
labeled_img[label_hue == 0] = 0

plt.subplot(1,1,1)
plt.title('Objects counted:'+ str(ret-1))
plt.imshow(labeled_img)
print('objects number is:', ret-1)
plt.show()

print('ok')


