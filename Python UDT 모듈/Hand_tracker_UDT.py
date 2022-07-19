import cv2
import mediapipe as mp
import socket
from utils import *

UDP_IP_ADDRESS = "127.0.0.1"
UDP_PORT_NO = 22222
Message = "0"
clientSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Initialize the mediapipe drawing class.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands_video = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                             min_detection_confidence=0.7, min_tracking_confidence=0.4)


camera_video = cv2.VideoCapture(0)
camera_video.set(3, 1920)
camera_video.set(4, 1080)
while(True):
    ret, frame = camera_video.read()
    #frame = cv2.convertScaleAbs(frame, alpha=1, beta=-20)
    if not ret:

        continue

    # frame = cv2.flip(frame, -1)

    # Perform Hands landmarks detection.
    frame, results = detectHandsLandmarks(frame, hands_video, display=False)
    try:
        if results.multi_hand_landmarks:

            # Perform hand(s) type (left or right) classification.
            _, hands_status, hand_type_ = getHandType(
                frame.copy(), results, draw=False, display=False)
            # Draw bounding boxes around the detected hands and write their classified types near them.
            frame, output, x1, y1, x2, y2 = drawBoundingBoxes(
                frame, results, hands_status, display=False)

            # print(hand_type_, x1, y1, x2, y2)
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            frame = cv2.line(frame, (int(cx), int(cy)),
                             (int(cx), int(cy)), (0, 0, 255), 5)

            print("cx val >> ", cx, "cx type >> ", type(cx))
            cx_int = int(cx)
            cy_int = int(cy)
            cx_str = str(cx_int)
            cy_str = str(cy_int)
            print("chagen cx type >> ", type(cx_str))

            print("message >> ", cx_str + cy_str)
            # Message = str(str(cx) + str(cy))
            Message = cx_str + cy_str
            print("cx >> ", int(int(Message)/1000))
            print("cy >> ", int(Message) % 1000)

        # Message = str(str(cx) + str(cy))
        # print("cx >> ", int(int(Message)/1000))
        # print("cy >> ", int(Message) % 1000)

        clientSock.sendto(bytes(Message, 'utf-8'),
                          (UDP_IP_ADDRESS, UDP_PORT_NO))
    except:
        clientSock.sendto(bytes(Message, 'utf-8'),
                          (UDP_IP_ADDRESS, UDP_PORT_NO))
        # #print("none exist")

    # frame = cv2.putText(hsv, str(str(cx)+","+str(Message)),
    #                     (cx, cy), font,  fontScale, color, thickness, cv2.LINE_AA)
    # cv2.imshow("frame", frame)
    cv2.imshow('Hands Landmarks Detection', frame)

    cv2.waitKey(1)
