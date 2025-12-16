# Real-Time CNN-Based Face Liveness and Anti-Spoofing System

## 📌 Project Overview
This project implements a real-time hybrid face liveness detection system to prevent spoofing attacks such as printed photos and video replays. The system combines deep learning-based classification with physiological blink detection to ensure reliable and secure face authentication.

A Convolutional Neural Network (CNN) based on the EfficientNetB0 architecture is used to classify faces as real or fake, while Eye Aspect Ratio (EAR) based blink detection using MediaPipe FaceMesh verifies natural eye movement in real time.

## 🚀 Key Features
- Real-time face liveness detection using webcam
- CNN-based spoof detection (EfficientNetB0)
- Eye blink detection using EAR and MediaPipe FaceMesh
- Robust against photo and video replay attacks
- Streamlit-based interactive web application
- Achieves 96.5% validation accuracy

## 🧠 Technologies Used
- Python  
- TensorFlow / Keras  
- EfficientNetB0  
- OpenCV  
- MediaPipe FaceMesh  
- Streamlit  
- NumPy  

## ⚙️ System Workflow
1. Capture live video input from webcam
2. Detect facial landmarks using MediaPipe FaceMesh
3. Calculate Eye Aspect Ratio (EAR) for blink detection
4. Classify face as Real or Fake using CNN
5. Final decision made using hybrid verification
6. Display result in real time using Streamlit

## 📸 Demo Results

### ✅ Real Face Detection
![Real Face](demo/image_name.png)

### ❌ Fake Face Detection
![Fake Face](demo/fake_image.png)


## ▶️ How to Run the Project

### 1️⃣ Clone the repository
```bash
git clone https://github.com/anjanapatlolla/Real-time-CNN-based-face-liveness-anti-spoofing-system.git
cd Real-time-CNN-based-face-liveness-anti-spoofing-system
