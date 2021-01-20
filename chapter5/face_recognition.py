import os
import cv2
import numpy
import pickle

def read_images(path, image_size):
    names = []
    training_images, training_labels = [], []
    label = 0
    for dirname, subdirnames, filenames in os.walk(path):
        for subdirname in subdirnames:
            names.append(subdirname)
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                img = cv2.imread(os.path.join(subject_path, filename), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, image_size)
                training_images.append(img)
                training_labels.append(label)
            label += 1
    training_images = numpy.asarray(training_images, numpy.uint8)
    training_labels = numpy.asarray(training_labels, numpy.int32)
    return names, training_images, training_labels

path_to_training_images = './data/at'
training_image_size = (200, 200)
names, training_images, training_labels = read_images(path_to_training_images, training_image_size)

model_path = './trained.sav'
model = cv2.face.EigenFaceRecognizer_create()
model.train(training_images, training_labels)

face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')

videoPath = './test.mp4'
videoCapture = cv2.VideoCapture(videoPath)

while (cv2.waitKey(1) == -1):
    # Iterating each video frame
    success, frame = videoCapture.read()
    if success:
        # detect faces from the grayscale img
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)
        for x, y, w, h in faces:
            # draw a rectangle around detected faces
            cv2.rectangle(frame, (x,y), (x + w, y + h), (0, 255, 0), 2)
            # convert the frame to grayscale (for easier face detection)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # crop the face area of the grayscale img. We may crop the top half of the img for better eyes detection
            roi_gray = gray[y:y+h, x:x+w]
            # The ROI is empty. May be the face is at the image edge
            if roi_gray.size == 0:
                continue
            roi_gray = cv2.resize(roi_gray, training_image_size)
            # Predict labels and confidence level
            label, confidence = model.predict(roi_gray)

            # Put a text to classify a face
            text = '%s, confidence=%.2f' % (names[label], confidence)
            cv2.putText(frame, text, (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Face Recognition', frame)