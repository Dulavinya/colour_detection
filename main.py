import cv2
from PIL import Image
from init import get_limits


yellow = (0, 255, 255) #yellow in BGR colourspace
green = (0, 255, 0)  # green in BGR colourspace
blue = (255, 0, 0)   # blue in BGR colourspace

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lowerLimit, upperLimit =get_limits(color=yellow)  
    lowerLimitGreen, upperLimitGreen = get_limits(color=green)
    lowerLimitBlue, upperLimitBlue = get_limits(color=blue)


    mask =cv2.inRange(hsvImage,lowerLimit, upperLimit)
    mask_green = cv2.inRange(hsvImage, lowerLimitGreen, upperLimitGreen)
    mask_blue = cv2.inRange(hsvImage, lowerLimitBlue, upperLimitBlue)


    mask_ = Image.fromarray(mask)
    mask_green_ = Image.fromarray(mask_green)
    mask_blue_ = Image.fromarray(mask_blue)


    bbox = mask_.getbbox()
    bbox_green = mask_green_.getbbox()
    bbox_blue = mask_blue_.getbbox()

    if bbox_green is not None:
        x1, y1, x2, y2 = bbox_green
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "green", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if bbox_blue is not None:
        x1, y1, x2, y2 = bbox_blue
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, "blue", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, "yellow", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
       

    print(bbox, bbox_green, bbox_blue)
    
    
    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()

