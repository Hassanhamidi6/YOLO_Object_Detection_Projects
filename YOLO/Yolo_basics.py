from ultralytics import YOLO
import cv2 

model = YOLO('../Yolo-weights/yolov8l.pt')
results = model("YOLO\images\Image1.jpg", show= True)

image_with_boxes =results[0].plot()

cv2.imshow("Detections", image_with_boxes )
cv2.waitKey(0)
cv2.destroyAllWindows()