import cv2
import mediapipe as mp
import time
 
cap = cv2.VideoCapture(0)
 
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

velocitys = [0]*21

while True:
	success, img = cap.read()
	imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	results = hands.process(imgRGB)

	# for i in range(10):
	# 	cv2.rectangle(img,
    #           	pt1=(50*i, 350),
    #           	pt2=(50*(i+1), 500),
    #           	color=(255, 255, 255),
    #           	thickness=-1,
    #           	lineType=cv2.LINE_4,
    #           	shift=0)
 
	if results.multi_hand_landmarks:
		for handLms in results.multi_hand_landmarks:
			for id, lm in enumerate(handLms.landmark):
				print(id)
				print(lm)
				h, w, c = img.shape
				cx, cy = int(lm.x * w), int(lm.y * h)
			
			mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
 
	cv2.putText(img, "Hand Tracking Test", (10, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
 
	cv2.imshow("Image", img)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break