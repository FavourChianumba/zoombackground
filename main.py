import time
import cv2
import cv2.cv2
import mediapipe as mp
import numpy as np
import os

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation

BG_COLOUR = (0, 255, 0)
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
prevTime = 0

listImg = os.listdir("images")
imgList = []
for imgPath in listImg:
    img = cv2.imread(f'images/{imgPath}')
    imgList.append(img)
indexImg = 0

#Webcam Input

with mp_selfie_segmentation.SelfieSegmentation(model_selection=0) as selfie_segmentation:
    bg_image = None
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame")
            continue

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = selfie_segmentation.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.9

        #Apply some background

        bg_image = cv2.imread("images/1.png")  #create virtual background
        #bg_image = cv2.GaussianBlur(image,(55,55),0) #Blur Background

        if bg_image is None:
            bg_image = np.zeros(image.shape, dtype=np.uint8)
            bg_image[:] = BG_COLOUR
        out_image = np.where(condition, image, imgList[indexImg])

        #retrn elements chosen from x or y depending on condition

        #Get frame rate

        current_Time = time.time()
        FPS = 1/(current_Time-prevTime)
        prevTime = current_Time

        cv2.putText(out_image, f'FPS: {int(FPS)}', (20, 70), cv2.cv2.FONT_ITALIC, 3, (0, 123, 213), 2)

        cv2.imshow('DIY virtual background', out_image)
        key = cv2.waitKey(5)
        if key == ord('a'):
            if indexImg > 0:
                indexImg -= 1
        elif key == ord('d'):
            if indexImg < len(imgList) - 1:
                indexImg += 1
        elif key & 0xFF == 27:
            break
cap.release()


