from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap = cv2.VideoCapture("Videos/cars.mp4")

# using  the YOLOv8 model
model = YOLO("../Yolo-weights/Yolov8n.pt")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 906)     # setting the video size 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 344)


# SORT tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Vehicle count tracking
total_count = 0
counted_ids = set()

# Line for counting the vehicles
limits = [0, 500, 1366, 500]  # (x1, y1, x2, y2)

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

            # Filter for vehicles
            if current_class in ["car", "truck", "motorbike", "bus"] and conf > 0.3:
                cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), l=9, rt=2, colorR=(255, 0, 255))
                # cvzone.putTextRect(img, f"{current_class} {conf}", (max(0, x1), max(35, y1)),
                #                    scale=0.8, thickness=1, offset=3)

                # Prepare for SORT tracking
                detections = np.vstack((detections, np.array([x1, y1, x2, y2, conf])))

    # SORT tracking
    results_tracker = tracker.update(detections)

    # Draw tracking and counting
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    for result in results_tracker:
        x1, y1, x2, y2, id = map(int, result)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2    # (centre x and centre y ) centre of the car rectangle

        cvzone.putTextRect(img, f"{current_class} {id}", (x1, y1), scale=0.8, thickness=1, offset=3)
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1] - 20 < cy < limits[3] + 20:  
            if id not in counted_ids:                # counting the unique cars when cross the line 
                counted_ids.add(id)
                total_count += 1
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5) # changing line color when car detected 

    # Display total count
    cvzone.putTextRect(img, f"Count: {total_count}", (50, 50), scale=1, thickness=2, offset=5)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
