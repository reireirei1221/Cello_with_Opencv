import cv2
import mediapipe as mp
import time
import subprocess
import copy
import numpy as np
import math

black = [0, 0, 0]
white = [255, 255, 255]

change_size = True

g_h = 480
g_w = 640

def almost_white_to_white(img):
	h, w, c = img.shape
	for i in range(h):
		for j in range(w):
			if img.item(i, j, 0) > 220 and img.item(i, j, 0) > 220 and img.item(i, j, 0) > 220:
				img.itemset((i,j,0),255)
				img.itemset((i,j,1),255)
				img.itemset((i,j,2),255)
	return img

# 弓とチェロの画像
original_cello_bow_img = cv2.imread("cello_bow.jpg")
original_cello_img = cv2.imread("cello.jpg")
original_cello_img = almost_white_to_white(original_cello_img)
original_cello_bow_img = almost_white_to_white(original_cello_bow_img)

cap = cv2.VideoCapture(0)

mpDraw = mp.solutions.drawing_utils

if change_size:
	mpPose = mp.solutions.pose
	pose = mpPose.Pose()
	time.sleep(3)
	success_pre, img_pre = cap.read()
	img_pre = cv2.flip(img_pre, 1)
	imgRGB_pre = cv2.cvtColor(img_pre, cv2.COLOR_BGR2RGB)
	results_pre = pose.process(imgRGB_pre)
	mpDraw.draw_landmarks(img_pre, results_pre.pose_landmarks, mpPose.POSE_CONNECTIONS)
	# print(results.pose_landmarks.landmark[11].x * 600, 11)
	# print(results.pose_landmarks.landmark[12].x * 600, 12)
	x_diff = (results_pre.pose_landmarks.landmark[11].x - results_pre.pose_landmarks.landmark[12].x)*g_w
	x_center = (results_pre.pose_landmarks.landmark[11].x + results_pre.pose_landmarks.landmark[12].x)*g_w/2
	y_center = (results_pre.pose_landmarks.landmark[11].y + results_pre.pose_landmarks.landmark[12].y)*g_h/2
	cello_diff = 320
	x_cello = 640*x_diff/cello_diff
	y_cello = 500*x_diff/cello_diff
	cv2.imshow("cello0", cv2.resize(original_cello_img, (600, 600)))
	cello_img_pre = cv2.resize(original_cello_img, (int(1200*x_diff/cello_diff), int(1200*x_diff/cello_diff)))
	cv2.imshow("cello1", cello_img_pre)
	M_pre = np.float32([[1,0,int(x_center-x_cello)],[0,1,int(y_center-y_cello)]])
	cv2.imshow("cello", cello_img_pre)
	resized_cello_img = cv2.warpAffine(cello_img_pre, M_pre, (g_w, g_h), borderValue=(255, 255, 255))
	cv2.imshow("img_pre", img_pre)
	# 駒の位置
	cx_cello = int(600*x_diff/cello_diff)+int(x_center-x_cello)
	cy_cello = int(720*x_diff/cello_diff)+int(y_center-y_cello)
	# +int(x_center-x_cello)
	# +int(y_center-y_cello)
	print(cx_cello, cy_cello)
	# 弓のresize
	h_bow, w_bow, c_bow = original_cello_bow_img.shape
	cello_bow_img = np.ones((g_h, g_w, 3), np.uint8)*255
	#h_bow = int(h_bow * x_diff/cello_diff)
	#w_bow = int(w_bow * x_diff/cello_diff)
	#print(w_bow, h_bow)
	#cello_bow_img[cy_cello-h_bow//2:cy_cello+h_bow//2, cx_cello-w_bow//4:cx_cello+w_bow-w_bow//4] = cv2.resize(original_cello_bow_img, (w_bow, h_bow))
	if cy_cello < 65:
		cy_cello = 65
	elif cy_cello+55 > g_h:
		cy_cello = g_h - 55
	if cx_cello < 136:
		cx_cello = 136
	elif cx_cello + 376 > g_w:
		cx_cello = g_w - 376
	cello_bow_img[cy_cello-65:cy_cello+55, cx_cello-136:cx_cello+376] = original_cello_bow_img

# チェロのresize
if not change_size:
	# 駒の位置
	cx_cello = 200
	cy_cello = 300
	resized_cello_img = original_cello_img[300:900, 320:1120]
	resized_cello_img = cv2.resize(resized_cello_img, (g_w, g_h))
	resized_cello_img = almost_white_to_white(resized_cello_img)
	# resized_cello_img[0:200, 120:320] = cv2.resize(resized_cello_img[50:150, 170:270], (200, 200))
	cv2.imshow('resized', resized_cello_img)
	# 弓のresize
	cello_bow_img = np.ones((g_h, g_w, 3), np.uint8)*255
	cello_bow_img[295+cy_cello-360:415+cy_cello-360, 64+cx_cello-200:576+cx_cello-200] = original_cello_bow_img

# チェロのマスクと画像生成
cello_msk_img = copy.deepcopy(resized_cello_img)
cello_img = copy.deepcopy(resized_cello_img)
cello_msk_img[np.where((resized_cello_img != white).all(axis=2))] = black
cello_img = cello_msk_img ^ resized_cello_img
# cello_img = almost_white_to_white(cello_img)
# cv2.imshow('masked_cello', cello_msk_img)
# cv2.imshow('cello', cello_img)

# # 弓のresize
# cello_bow_img = np.ones((g_h, g_w, 3), np.uint8)*255
# cello_bow_img[295+cy_cello-360:415+cy_cello-360, 64+cx_cello-200:576+cx_cello-200] = original_cello_bow_img
# cello_bow_img = almost_white_to_white(cello_bow_img)

# cv2.rectangle(cello_bow_img, (544, 252), (585, 273), (255, 255, 255), thickness=-1)

# 弓のマスクと画像生成
cello_bow_msk_img = copy.deepcopy(cello_bow_img)
cello_bow_msk_img[np.where((cello_bow_img != white).all(axis=2))] = black
cello_bow_img = cello_bow_msk_img ^ cello_bow_img

cv2.imshow('bow', cello_bow_img)
 
# 本編開始
 
mpHands = mp.solutions.hands
hands = mpHands.Hands()
# mpDraw = mp.solutions.drawing_utils

id1 = 8
id2 = 12
id3 = 9 # 左手のポジション

# 弓の長さ
bow_length = 400
# プロセス
pro = None
# setting.pro1 = None
# 弓が動いてる方向のbool
move_toward_right = False
move_toward_left = False
# 音を出すbool
make_sound = False
# 前の手の位置
pre_distance = None
pre_cx_hand = None
pre_cy_hand = None

string = None
pre_string = None

position = 0
pre_position = None

cx_hand = None
cy_hand = None

num_finger = 0
pre_numfinger = None

flag_terminate = False

off_terminate = False

value1 = 2
value2 = 5
value3 = 1.1

def find_which_string(x, y):
	x_ = x-cx_cello
	y_ = -(y-cy_cello)
	if x_ < 0:
		return 0
	if value1*x_ <= value2*y_:
		return 'A'
	elif 0 <= y_:
		return 'D'
	elif -value1*x_ <= value2*y_:
		return 'G'
	else:
		return 'C'

def find_position(y):
	if y*g_h < value3*cy_cello//3:
		return 0
	elif y*g_h < value3*(cy_cello//3)*2:
		return 1
	else:
		return 2
	
while True:
	
	success, img = cap.read()
	img = cv2.flip(img, 1)
	imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	results = hands.process(imgRGB)
	# results_pose = pose.process(imgRGB)

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

			rightHand = (cx_cello + 100 < handLms.landmark[5].x * w)
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
				cx_5 = handLms.landmark[id1].x * w
				cy_5 = handLms.landmark[id1].y * h
				cx_17 = handLms.landmark[id2].x * w
				cy_17 = handLms.landmark[id2].y * h

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
				#if cx_hand == cx_cello:
					#break
				gradient = (cy_hand-cy_cello)/(cx_hand-cx_cello)
				degree = math.degrees(math.atan(gradient))
				mat_rotate = cv2.getRotationMatrix2D((cx_cello, cy_cello), -degree, 1)
				rotated_cello_bow_img = cv2.warpAffine(moved_cello_bow_img, mat_rotate, (g_w, g_h), borderValue=(0, 0, 0))
				rotated_cello_bow_msk_img = cv2.warpAffine(moved_cello_bow_msk_img, mat_rotate, (g_w, g_h), borderValue=(255, 255, 255))

				img = img & rotated_cello_bow_msk_img
				img = img ^ rotated_cello_bow_img

				# img = cv2.line(img, (cx_hand, cy_hand), (cx_bow, cy_bow), (116, 80, 48), 10)
				if pre_distance is not None:
					if (abs(distance - pre_distance) > 5):
						if cx_hand-pre_cx_hand > 5:
							if move_toward_left:
								move_toward_left = False
								make_sound = True
								if  handLms.landmark[4].x > handLms.landmark[9].x:
									off_terminate = True
							move_toward_right = True
						elif cx_hand-pre_cx_hand < -5:
							if move_toward_right:
								move_toward_right = False
								make_sound = True
								if  handLms.landmark[4].x > handLms.landmark[5].x:
									off_terminate = True
							move_toward_left = True
						else:
							make_sound = False
				
					
					# if (cx_hand-pre_cx_hand < 0) and (abs(distance - pre_distance) > 5):
					# 	make_sound = True
					# 	if move_toward_right:
					# 		move_toward_left = True
					# 		move_toward_right = False
					# 		# pro = subprocess.Popen(['python3', 'sub.py', 'violin'])
					# 	elif not move_toward_left:
					# 		move_toward_left = True
					# 		# pro = subprocess.Popen(['python3', 'sub.py', 'violin'])

				# if string != find_which_string(cx_hand, cy_hand):
				# 	string = find_which_string(cx_hand, cy_hand)
				# 	if pro != None:
				# 		pro.terminate()
				# 	pro = subprocess.Popen(['python3', 'sub.py', 'violin'])
				# 	print(string)
				
				pre_distance = distance
				pre_cx_hand = cx_hand
				pre_cy_hand = cy_hand

			else:

				num_finger = 0
				for i in range(4):
					second_joint = 4*(i+1) + 2
					finger_top = 4*(i+1) + 4
					cx_second = handLms.landmark[second_joint].x * w
					cx_top = handLms.landmark[finger_top].x * w

					# 指が伸びていたら
					if cx_second + 10 < cx_top:
						break

					num_finger += 1
				if pre_numfinger != num_finger:
					make_sound = True
					pre_numfinger = num_finger

				position = find_position(handLms.landmark[id3].y)
				if pre_position != position:
					make_sound = True
					pre_position = position
				# print(num_finger, position)
			

		if pro != None:
			if cx_hand != None:
				string = find_which_string(cx_hand, cy_hand)
				if pre_string != string:
					make_sound = True
					pre_string = string

		if make_sound and cx_hand != None:
			string = find_which_string(cx_hand, cy_hand)
			if pro != None:
				off_terminate = False
				if not off_terminate:
					pro.terminate()
			if string != 0:
				if num_finger == 0:
					pro = subprocess.Popen(['python3', 'sub.py', string + str(0), '>', '/dev/null', '2>&1'])
				else:
					pro = subprocess.Popen(['python3', 'sub.py', string + str(4*position+num_finger), '>', '/dev/null', '2>&1'])

		make_sound = False
		off_terminate = False
	
	num_finger = 0

	img = cv2.line(img, (cx_cello+100*value2, cy_cello-100*value1), (cx_cello, cy_cello), (255, 255, 255), 1)
	img = cv2.line(img, (cx_cello+100*value2, cy_cello+100*value1), (cx_cello, cy_cello), (255, 255, 255), 1)
	img = cv2.line(img, (cx_cello, cy_cello), (cx_cello+100*value2, cy_cello), (255, 255, 255), 1)
	img = cv2.line(img, (cx_cello-100*value2, int(value3*cy_cello//3)), (cx_cello, int(value3*cy_cello//3)), (255, 255, 255), 1)
	img = cv2.line(img, (cx_cello-100*value2, int(value3*2*cy_cello//3)), (cx_cello, int(value3*2*cy_cello//3)), (255, 255, 255), 1)

	# cv2.putText(img, "Hand Tracking Test", (10, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

	img = cv2.resize(img, (1200, 900))

	cv2.imshow("Image", img)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		if pro != None:
			pro.terminate()
		break
