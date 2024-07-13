import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

face_casecade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

model_path = 'C:/Users/susdh/Desktop/Projects ms/DeepLearning_workspace/Deep-learning-lab/Image-Project/model.h5'
emotion_classifier = load_model(model_path)

EMOTIONS = ["Angry", "Disgust", "Fear", "happy", "neutral", "Sad", "surprise"]

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error opening video")
    exit(1)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        continue

    labels = []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_casecade.detectMultiScale(
        gray, scaleFactor=1.3,  minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype("float")/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = emotion_classifier.predict(roi)[0]
            label = EMOTIONS[prediction.argmax()]
            label_position = (x, y)
            cv2.putText(frame, label, label_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No Faces', (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
