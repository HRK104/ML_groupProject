# importing the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import imutils
import cv2

face_detector = cv2.dnn.readNet('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')
mask_detector = load_model('detect_mask.model')
test_video = cv2.VideoCapture('test_video.mp4')

while True:
    ret, frame = test_video.read()
    frame = imutils.resize(frame, width=400)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_detector.setInput(blob)
    detections = face_detector.forward()

    detected_faces = []
    bonding_boxes = []
    detection_results = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

            # returns the integer value of the start and end of the bounding box
            (startX, startY, endX, endY) = box.astype("int")
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            # convert the face into an array
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            # append the face and bounding box to their respective lists
            detected_faces.append(face)
            bonding_boxes.append((startX, startY, endX, endY))

    if len(detected_faces) > 0:
        detection_results = mask_detector.predict(detected_faces)

    for (face_box, result) in zip(bonding_boxes, detection_results):
        (startX, startY, endX, endY) = face_box
        (mask, withoutMask) = result

        if mask > withoutMask:
            label = "With Mask"
            color = (0, 255, 0)
        else:
            label = "Without Mask"
            color = (0, 0, 255)

        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("x"):
        break
