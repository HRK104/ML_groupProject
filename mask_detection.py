


import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import imutils
from tensorflow.keras import regularizers
from keras.layers import Conv2D
from tensorflow import keras
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from sklearn.dummy import DummyClassifier




face_detector = cv2.dnn.readNet('face_detector/deploy.prototxt', 'face_detector/res10_300x300_ssd_iter_140000.caffemodel')
mask_detector = load_model('detect_mask.model')
test_video = cv2.VideoCapture('test_video.mov')

while True:
    detected_faces = []
    bonding_boxes = []
    detection_results = []
    
    others, frame = test_video.read()
    frame = imutils.resize(frame, width=500)
    (height, width) = frame.shape[:2]
    settings = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (100.0, 200.0, 125.0))
    face_detector.setInput(settings)
    detects = face_detector.forward()

    for i in range(0, detects.shape[2]):
        confidence = detects[0, 0, i, 2]
        if confidence > 0.6:
            box = detects[0, 0, i, 3:7] * np.array([width, height, width, height])

            # returns the integer value of the start and end of the bounding box
            (X_start, Y_start, X_end, Y_end) = box.astype("int")
            detecded_face = frame[Y_start:Y_end, X_start:X_end]
            detecded_face = cv2.cvtColor(detecded_face, cv2.COLOR_BGR2RGB)
            detecded_face = cv2.resize(detecded_face, (255, 255))
            # convert the face into an array
            detecded_face = img_to_array(detecded_face)
            detecded_face = preprocess_input(detecded_face)
            detecded_face = np.expand_dims(detecded_face, axis=0)

            # append the face and bounding box to their respective lists
            detected_faces.append(detecded_face)
            bonding_boxes.append((X_start, Y_start, X_end, Y_end))

    if len(detected_faces) > 0:
        detection_results = mask_detector.predict(detected_faces)

    for (face_box, result) in zip(bonding_boxes, detection_results):
        (X_start, Y_start, X_end, Y_end) = face_box
        (mask, withoutMask) = result

        if mask > withoutMask:
            label = "With Mask"
            color = (100, 255, 0)
        else:
            label = "Without Mask"
            color = (100, 0, 255)

        cv2.putText(frame, label, (X_start, Y_start - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.rectangle(frame, (X_start, Y_start), (X_end, Y_end), color, 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("x"):
        break