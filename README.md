# 🚦 Object Detection & Tracking with YOLOv8

This repository contains two real-world computer vision projects built using **YOLOv8** (for object detection) and **SORT** (for object tracking).  

Both projects give every detected object a **unique identity**, track it across video frames, and **count objects when they cross a virtual line**.  

---

## 📌 Projects Included

### 1️⃣ Car Counter 🚗

![App Screenshot](Project1-CarCounter\screenshot.PNG)

- Detects and tracks cars in a video stream  
- Each car is assigned a **unique ID**  
- A **counting line** is drawn → whenever a car crosses the line, the count increases by 1  
- Helps in **traffic monitoring, smart city planning, and vehicle analysis**  

👉 Example use case: Counting cars on a highway or road for traffic analysis.  

---

### 2️⃣ Person Counter 🧑‍🤝‍🧑

![App Screenshot](Project2-PersonCounter\screenshot.PNG)

- Detects and tracks people in real-time  
- Each person gets a **unique ID**  
- Special setup for **escalators**:
  - One line for people going **up**  
  - Another line for people going **down**  
- Counts how many people cross each line  

👉 Example use case: Counting how many peoples come and go at any place .  

---

## 🔑 Key Features
- ✅ **YOLOv8** → Fast & accurate object detection  
- ✅ **SORT Algorithm** → Smooth tracking with unique IDs  
- ✅ **Line Crossing Logic** → Easy counting mechanism  
- ✅ **Real-time Performance** → Works with live video feeds or recorded footage  
- ✅ **Customizable** → Can be adapted for other objects (bikes, buses, animals, etc.)  

---

## ⚙️ How It Works
1. **Detection:** YOLOv8 detects cars or persons in each frame  
2. **Tracking:** SORT assigns a unique ID to each object  
3. **Counting:** When an object crosses a predefined line → counter +1  

---

## 🛠️ Tech Stack
- **Python** 🐍  
- **YOLOv8 (Ultralytics)** for object detection  
- **SORT (Simple Online and Realtime Tracking)** for tracking  
- **OpenCV** for video processing  

---

## 📸 Example Output (Concept)

