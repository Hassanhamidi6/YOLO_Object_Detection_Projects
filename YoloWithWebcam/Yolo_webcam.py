from ultralytics import YOLO
import cv2
import cvzone
import math

cap = cv2.VideoCapture(0)    

model = YOLO("../Yolo-weights/Yolov8n.pt")

classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella","handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat","baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup","fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli","carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed","diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone","microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors","teddy bear", "hair drier", "toothbrush"
]

 
while True:
    success, img = cap.read()
    if not success:
        print("No frame received. Exiting...")
        break

    results = model(img, stream=True)
    for r in results:
        boxes= r.boxes
        for box in boxes:

            # Bounding Box
            x1, y1, x2, y2 =box.xyxy[0]
            x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
            # print(x1, y1, x2, y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,255), 3)

            w, h =x2-x1, y2-y1
            cvzone.cornerRect(img, (x1,y1,w,h)) 

            #Confidence
            conf = math.ceil((box.conf[0]*100))/100
            print(conf)
            # cvzone.putTextRect(img, f"{conf}", (max(0, x1), max(35, y1))) 

            # Class name
            cls = int(box.cls[0])
            current_class= classNames[cls]

            # Here we are giving the name and confidence to the object  
            cvzone.putTextRect(img, f"{current_class} {conf}", (max(0,x1), max(35,y1)),
                               scale=0.7, thickness=1, offset=5)


    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
 