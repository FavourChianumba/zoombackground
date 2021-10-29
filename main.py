import time
import cv2
import cv2.cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation

#For static images:
IMAGE_FILES = []
BG_COLOR = (192, 192, 192) # gray
MASK_COLOR = (255, 255, 255) # white
with mp_selfie_segmentation.SelfieSegmentation(
    model_selection=0) as selfie_segmentation:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    image_height, image_width, _ = image.shape
    # Convert the BGR image to RGB before processing.
    results = selfie_segmentation.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Draw selfie segmentation on the background image.
    # To improve segmentation around boundaries, consider applying a joint
    # bilateral filter to "results.segmentation_mask" with "image".
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    # Generate solid color images for showing the output selfie segmentation mask.
    fg_image = np.zeros(image.shape, dtype=np.uint8)
    fg_image[:] = MASK_COLOR
    bg_image = np.zeros(image.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR
    output_image = np.where(condition, fg_image, bg_image)
    cv2.imwrite('/tmp/selfie_segmentation_output' + str(idx) + '.png', output_image)


BG_COLOUR = (0, 255, 0)
cap = cv2.VideoCapture(0)
prevTime = 0

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

        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.5

        #Apply some background

        bg_image = cv2.imread('backgrounds/1.png')  #create virtual background
        #bg_image = cv2.GaussianBlur(image,(55,55),0) #Blur Background

        if bg_image is None:
            bg_image = np.zeros(image.shape, dtype=np.uint8)
            bg_image[:] = BG_COLOUR
        out_image = np.where(condition, image, bg_image)

        #retrn elements chosen from x or y depending on condition

        #Get frame rate

        current_Time = time.time()
        FPS = 1/(current_Time-prevTime)
        prevTime = current_Time

        cv2.putText(out_image, f'FPS: {int(FPS)}', (20, 70), cv2.cv2.FONT_ITALIC, 3, (0, 123, 213), 2)

        cv2.imshow('DIY virtual background', out_image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()


