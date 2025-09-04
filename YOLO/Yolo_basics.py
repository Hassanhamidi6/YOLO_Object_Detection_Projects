from ultralytics import YOLO
import cv2 

# Yolo have many versions like "large", "Small", "Medium", "Nano"
model = YOLO('../Yolo-weights/yolov8n.pt')  # here we are using yolo  nano
results = model("YOLO\images\cars.jpg", show= True)

image_with_boxes =results[0].plot()



cv2.imshow("Detections", image_with_boxes )
cv2.waitKey(0)
cv2.destroyAllWindows()