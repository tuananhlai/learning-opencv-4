import cv2

face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./cascades/haarcascade_eye.xml')

videoPath = './video.mp4'
videoCapture = cv2.VideoCapture(videoPath)

while (cv2.waitKey(1) == -1):
    # Iterating each video frame
    success, frame = videoCapture.read()
    if success:
        # convert the frame to grayscale (for easier face detection)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detect faces from the grayscale img
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(120, 120))
        for x, y, w, h in faces:
            # draw a rectangle around detected faces
            cv2.rectangle(frame, (x,y), (x + w, y + h), (0, 255, 0), 2)
            # crop the face area of the grayscale img. We may crop the top half of the img for better eyes detection
            roi_gray = gray[y:y+int(h/2), x:x+w]
            # detect eyes region from the cropped area of a face
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.03, 5, minSize=(40,40))
            for (ex, ey, ew, eh) in eyes:
                # draw rect around the eyes
                cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (255, 0, 0), 2)
        cv2.imshow('Face Detection', frame)