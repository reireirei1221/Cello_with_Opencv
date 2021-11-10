import cv2
import mediapipe as mp
import time
import subprocess
import copy
import numpy as np
import math

black = [0, 0, 0]
white = [255, 255, 255]

g_h = 480
g_w = 640

def almost_white_to_white(img):
	h, w, c = img.shape
	for i in range(h):
		for j in range(w):
			if img.item(i, j, 0) > 230 and img.item(i, j, 0) > 230 and img.item(i, j, 0) > 230:
				img.itemset((i,j,0),255)
				img.itemset((i,j,1),255)
				img.itemset((i,j,2),255)
	return img

# 弓とチェロの画像
original_cello_bow_img = cv2.imread("cello_bow.jpg")
original_cello_img = cv2.imread("cello.jpg")

# 弓のresize
cello_bow_img = np.ones((g_h, g_w, 3), np.uint8)*255
cello_bow_img[295:415, 64:576] = original_cello_bow_img
cello_bow_img = almost_white_to_white(cello_bow_img)

# cv2.rectangle(cello_bow_img, (544, 252), (585, 273), (255, 255, 255), thickness=-1)

# 弓のマスクと画像生成
cello_bow_msk_img = copy.deepcopy(cello_bow_img)
cello_bow_msk_img[np.where((cello_bow_img != white).all(axis=2))] = black
cello_bow_img = cello_bow_msk_img ^ cello_bow_img

# チェロのresize
resized_cello_img = original_cello_img[300:900, 320:1120]
resized_cello_img = cv2.resize(resized_cello_img, (640, 480))
resized_cello_img = almost_white_to_white(resized_cello_img)
# resized_cello_img[0:200, 120:320] = cv2.resize(resized_cello_img[50:150, 170:270], (200, 200))
# cv2.imshow('resized', resized_cello_img)

# チェロのマスクと画像生成
cello_msk_img = copy.deepcopy(resized_cello_img)
cello_img = copy.deepcopy(resized_cello_img)
cello_msk_img[np.where((resized_cello_img != white).all(axis=2))] = black
cello_img = cello_msk_img ^ resized_cello_img
# cv2.imshow('masked_cello', cello_msk_img)
# cv2.imshow('cello', cello_img)
 
# 本編開始
cap = cv2.VideoCapture(0)
 
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
# 駒の位置
cx_cello = 200
cy_cello = 360
# 弓の長さ
bow_length = 400
# プロセス
pro = None

string = None

def find_which_string(x, y):
	x_ = x-cx_cello
	y_ = -(y-cy_cello)
	if x_ < 0:
		return 0
	if 2*x_ <= 5*y_:
		return 1
	elif 0 <= y_:
		return 2
	elif -2*x_ <= 5*y_:
		return 3
	else:
		return 4
	
while True:
	success, img = cap.read()
	img = cv2.flip(img, 1)
	imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	results = hands.process(imgRGB)

	img = img & cello_msk_img

	img = img ^ cello_img

	# img = img & cello_bow_msk_img
	# img = img ^ cello_bow_img

	# # A線
	# img = cv2.circle(img, (250, 80), 10, (255,255,255), thickness=2)
	# img = cv2.circle(img, (250, 100), 10, (255,255,255), thickness=2)
	# img = cv2.circle(img, (250, 120), 10, (255,255,255), thickness=2)
	# img = cv2.circle(img, (250, 140), 10, (255,255,255), thickness=2)

	# # D線
	# img = cv2.circle(img, (230, 80), 10, (255,255,255), thickness=2)
	# img = cv2.circle(img, (230, 100), 10, (255,255,255), thickness=2)
	# img = cv2.circle(img, (230, 120), 10, (255,255,255), thickness=2)
	# img = cv2.circle(img, (230, 140), 10, (255,255,255), thickness=2)

	# # G線
	# img = cv2.circle(img, (210, 80), 10, (255,255,255), thickness=2)
	# img = cv2.circle(img, (210, 100), 10, (255,255,255), thickness=2)
	# img = cv2.circle(img, (210, 120), 10, (255,255,255), thickness=2)
	# img = cv2.circle(img, (210, 140), 10, (255,255,255), thickness=2)

	# # C線
	# img = cv2.circle(img, (190, 80), 10, (255,255,255), thickness=2)
	# img = cv2.circle(img, (190, 100), 10,(255,255,255), thickness=2)
	# img = cv2.circle(img, (190, 120), 10, (255,255,255), thickness=2)
	# img = cv2.circle(img, (190, 140), 10, (255,255,255), thickness=2)
 
	if results.multi_hand_landmarks:
		for handLms in results.multi_hand_landmarks:
			h, w, c = img.shape

			rightHand = (cx_cello + 50 < handLms.landmark[5].x * w)
			# rightHand = False

			# for id, lm in enumerate(handLms.landmark):
				# print(id)
				# print(lm)
				# h, w, c = img.shape
				# cx, cy = int(lm.x * w), int(lm.y * h)
				# if id == 5:
					# cx_5, cy_5 = cx, cy
				# if id == 17:
					# cx_17, cy_17 = cx, cy
			
			mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

			if rightHand:
				cx_5 = handLms.landmark[7].x * w
				cy_5 = handLms.landmark[7].y * h
				cx_17 = handLms.landmark[19].x * w
				cy_17 = handLms.landmark[19].y * h

				cx_hand = (cx_5+cx_17)/2
				cy_hand = (cy_5+cy_17)/2

				distance = ((cx_hand-cx_cello)**2 + (cy_hand-cy_cello)**2)**0.5

				cx_bow = (bow_length*cx_cello + (distance-bow_length)*cx_hand)/distance
				cy_bow = (bow_length*cy_cello + (distance-bow_length)*cy_hand)/distance

				# 弓の平行移動
				mat_move = np.float32([[1,0,distance-300],[0,1,0]])
				moved_cello_bow_img = cv2.warpAffine(cello_bow_img,mat_move,(g_w,g_h),borderValue=(0, 0, 0))
				moved_cello_bow_msk_img = cv2.warpAffine(cello_bow_msk_img,mat_move,(g_w,g_h),borderValue=(255, 255, 255))

				# 各座標のint型への変換
				cx_hand = int(cx_hand)
				cy_hand = int(cy_hand)
				cx_bow = int(cx_bow)
				cy_bow = int(cy_bow)

				# 弓の回転
				gradient = (cy_hand-cy_cello)/(cx_hand-cx_cello)
				degree = math.degrees(math.atan(gradient))
				mat_rotate = cv2.getRotationMatrix2D((cx_cello, cy_cello), -degree, 1)
				rotated_cello_bow_img = cv2.warpAffine(moved_cello_bow_img, mat_rotate, (g_w, g_h), borderValue=(0, 0, 0))
				rotated_cello_bow_msk_img = cv2.warpAffine(moved_cello_bow_msk_img, mat_rotate, (g_w, g_h), borderValue=(255, 255, 255))

				img = img & rotated_cello_bow_msk_img
				img = img ^ rotated_cello_bow_img

				# img = cv2.line(img, (cx_hand, cy_hand), (cx_bow, cy_bow), (116, 80, 48), 10)
				
				if string != find_which_string(cx_hand, cy_hand):
					string = find_which_string(cx_hand, cy_hand)
					if pro != None:
						pro.terminate()
					pro = subprocess.Popen(['python3', 'sub.py', 'violin'])
					print(string)

	img = cv2.line(img, (cx_cello+500, cy_cello-200), (cx_cello-500, cy_cello+200), (255, 255, 255), 1)
	img = cv2.line(img, (cx_cello+500, cy_cello+200), (cx_cello-500, cy_cello-200), (255, 255, 255), 1)
	img = cv2.line(img, (cx_cello-500, cy_cello), (cx_cello+500, cy_cello), (255, 255, 255), 1)

	# img = cv2.rectangle(img, (cx_cello-50, 20), (cx_cello+50, cy_cello-50), (0, 0, 0), -1)

	# img = cv2.line(img, (cx_cello-10, 0), (cx_cello-10, cy_cello+50), (255, 255, 255), 2)


	# cv2.putText(img, "Hand Tracking Test", (10, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
	

	cv2.imshow("Image", img)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break