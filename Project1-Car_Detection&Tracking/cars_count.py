from ultralytics import YOLO
import cv2
import cvzone
from collections import defaultdict

# Load model
model = YOLO("../Yolo-weights/yolov8s.pt")

cap = cv2.VideoCapture("Videos\carsss.mp4")

# Counting
total_count = 0
counted_ids = set()
track_history = defaultdict(list)

# Counting line

limits = [500, 450, 1250, 450] # x1, y1, x2, y2

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.resize(img, (1920, 1080))

    # ByteTrack tracking
    results = model.track(img, conf=0.35, iou=0.5, persist=True, tracker="bytetrack.yaml")

    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0,255,255), 4)

    for r in results:
        if r.boxes.id is None:
            continue

        for box, track_id, cls in zip(r.boxes.xyxy, r.boxes.id, r.boxes.cls):
            x1,y1,x2,y2 = map(int, box)
            track_id = int(track_id)
            cls = int(cls)

            name = model.names[cls]

            if name not in ["car", "truck", "bus", "motorbike"]:
                continue

            cx, cy = (x1+x2)//2, (y1+y2)//2
            track_history[track_id].append(cy)

            if len(track_history[track_id]) >= 2:
                prev_y = track_history[track_id][-2]

                if prev_y < limits[1] and cy >= limits[1]:
                    if track_id not in counted_ids:
                        counted_ids.add(track_id)
                        total_count += 1

            cvzone.cornerRect(img, (x1,y1,x2-x1,y2-y1), l=9, rt=2)
            cvzone.putTextRect(img, f"{name}", (x1,y1), scale=0.6, thickness=1)
            cv2.circle(img, (cx,cy), 5, (0,255,255), cv2.FILLED)

    cvzone.putTextRect(img, f"Count: {total_count}",(80,80),scale=1.5,thickness=2, colorT=(0,255,255), font=cv2.FONT_HERSHEY_COMPLEX,)

    cv2.imshow("Traffic Counter", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
