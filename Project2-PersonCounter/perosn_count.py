from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap = cv2.VideoCapture("Videos/people.mp4") # capturing the video

model = YOLO("../Yolo-weights/Yolov8n.pt")    # using  the YOLOv8 model

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 906)   
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 344)


# SORT tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Line for counting the vehicles
limitsUp = [80, 150, 300, 150]  # (x1, y1, x2, y2)
limitsDown = [500, 550, 750, 550]  

# Vehicle count tracking
total_countUp = []
total_countDown = []

# Class names from COCO dataset
classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "sofa", "potted plant", "bed", "dining table", "toilet", "tvmonitor", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

while True:
    success, img = cap.read()
    if not success:
        break

    # Detect objects
    results = model(img, stream=True, conf=0.3)

    detections = np.empty((0, 5))  # For SORT

    for r in results:
        boxes = r.boxes 
        for box in boxes:   
            # Bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Confidence
            conf = math.ceil(box.conf[0] * 100) / 100

            # Class name
            cls = int(box.cls[0])
            current_class = classNames[cls]

            # Applying filter for Persons
            if current_class == "person" and conf > 0.3:
                cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), l=9, rt=2, colorR=(255, 0, 255))
                # cvzone.putTextRect(img, f"{current_class} {conf}", (max(0, x1), max(35, y1)),
                #                    scale=0.8, thickness=1, offset=3)

                # Prepare for SORT tracking
                detections = np.vstack((detections, np.array([x1, y1, x2, y2, conf])))

    # SORT tracking
    results_tracker = tracker.update(detections)

    # Draw tracking and counting line 
    cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 0, 255), 5)
    cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 0, 255), 5)

    for result in results_tracker:
        x1, y1, x2, y2, id = map(int, result)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2   # (centre x and centre y ) centre of the person's rectangle 

        cvzone.putTextRect(img, f"{current_class} {id}", (x1, y1), scale=0.8, thickness=1, offset=3)
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # for up count 
        if limitsUp[0] < cx < limitsUp[2] and limitsUp[1] - 20 < cy < limitsUp[3] + 20:
            if total_countUp.count(id)==0:
                total_countUp.append(id)
                cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 255, 0), 5)

        #for down count 
        if limitsDown[0] < cx < limitsDown[2] and limitsDown[1] - 20 < cy < limitsDown[3] + 20:
            if total_countDown.count(id)==0:
                total_countDown.append(id)
                cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 255, 0), 5)

    # ===== Display total count as a simple GUI box =====
    # Background rectangle
    cv2.rectangle(img, (10, 10), (220, 130), (50, 50, 50), -1)  # dark background
    cv2.rectangle(img, (10, 10), (220, 130), (255, 255, 255), 2)  # white border

    # Title
    cv2.putText(img, "People Count", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,0.8, (255, 255, 255), 2)

    # Up count
    cv2.putText(img, f"Up   : {len(total_countUp)}", (20, 80),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (139, 195, 75), 2)

    # Down count
    cv2.putText(img, f"Down : {len(total_countDown)}", (20, 110),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 230), 2)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
 