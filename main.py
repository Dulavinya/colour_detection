import cv2
from init import get_limits


yellow = (0, 255, 255) #yellow in BGR colourspace

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lowerLimit, upperLimit =get_limits(color=yellow)



    mask =cv2.inRange(hsvImage,lowerLimit, upperLimit)
    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()

