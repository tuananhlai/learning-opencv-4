import cv2

clicked = False

def onMouse(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONUP:
        clicked = True

cameraCapture = cv2.VideoCapture(0)
cv2.namedWindow('myWindow')
cv2.setMouseCallback('myWindow', onMouse)

print('Showing camera feed. Click window or press any key to stop')
success, frame = cameraCapture.read()
while success and cv2.waitKey(1) == -1 and not clicked:
    cv2.imshow('myWindow', frame)
    success, frame = cameraCapture.read()

cv2.destroyWindow('myWindow')
cameraCapture.release()