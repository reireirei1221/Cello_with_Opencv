import cv2
import mediapipe as mp
import time
import subprocess
 
cap = cv2.VideoCapture(0)
 
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
cx_cello = 200
cy_cello = 300

bow_length = 1000

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
 
	if results.multi_hand_landmarks:
		for handLms in results.multi_hand_landmarks:
			for id, lm in enumerate(handLms.landmark):
				# print(id)
				# print(lm)
				h, w, c = img.shape
				cx, cy = int(lm.x * w), int(lm.y * h)
				if id == 5:
					cx_5, cy_5 = cx, cy
				if id == 17:
					cx_17, cy_17 = cx, cy
			
			mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

			cx_hand = int((cx_5+cx_17)/2)
			cy_hand = int((cy_5+cy_17)/2)

			distance = ((cx_hand-cx_cello)**2 + (cy_hand-cy_cello)**2)**0.5

			cx_bow = int((bow_length*cx_cello + (distance-bow_length)*cx_hand)/distance)
			cy_bow = int((bow_length*cy_cello + (distance-bow_length)*cy_hand)/distance)

			img = cv2.line(img, (cx_hand, cy_hand), (cx_bow, cy_bow), (255, 255, 255), 10)

			if string != find_which_string(cx_hand, cy_hand):
				string = find_which_string(cx_hand, cy_hand)
				subprocess.Popen(['python3', 'sub.py', 'piano'])
				print(string)

	img = cv2.line(img, (cx_cello+500, cy_cello-200), (cx_cello-500, cy_cello+200), (255, 255, 255), 5)

	img = cv2.line(img, (cx_cello+500, cy_cello+200), (cx_cello-500, cy_cello-200), (255, 255, 255), 5)

	img = cv2.line(img, (cx_cello-500, cy_cello), (cx_cello+500, cy_cello), (255, 255, 255), 5)

	# cv2.putText(img, "Hand Tracking Test", (10, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
 
	cv2.imshow("Image", img)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break