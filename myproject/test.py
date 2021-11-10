import cv2
import numpy as np

cello_img = cv2.imread("cello.jpg")

cello_bow_img = cv2.imread("cello_bow.jpg")

blank = np.ones((480, 640, 3), np.uint8)*255

blank[180:300, 64:576] = cello_bow_img

cv2.rectangle(blank, (544, 252), (585, 273), (255, 255, 255), thickness=-1)

cv2.imshow('balnk', blank)


mat = cv2.getRotationMatrix2D((640 / 2, 480 / 2), 45, 1)

affine_img = cv2.warpAffine(blank, mat, (640, 480), borderValue=(255, 255, 255))

cv2.imshow('affine', affine_img)

# black = [0, 0, 0]
# white = [255, 255, 255]
# cello_img[np.where((cello_img == white).all(axis=2))] = black

cello_img = cello_img[300:900, 200:1000]

cello_img = cv2.resize(cello_img, (640, 480))

cv2.imshow('cello', cello_img)
 
cap = cv2.VideoCapture(0)

while True:
	success, img = cap.read()
	img = cv2.flip(img, 1)
	imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	h, w, c = img.shape
	# img = cv2.bitwise_and(img, cello_img)
	img = ~img ^ ~affine_img

	cv2.imshow("Image", ~img)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

print(img.shape)