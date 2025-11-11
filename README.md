Ultra-accurate American Sign Language (ASL) recognition system using MediaPipe hand tracking and a deep CNN trained on the Sign-MNIST dataset. The project integrates real-time gesture detection, live webcam inference, and model performance visualization with custom graph generation.

Tags: ASL CNN TensorFlow MediaPipe Computer Vision Deep Learning Sign Language Recognition Python


#  ASL Recognition Ultra – Deep Learning + MediaPipe
This repository contains a **complete pipeline for American Sign Language (ASL) recognition** using **MediaPipe hand tracking** and a **Convolutional Neural Network (CNN)** trained on the **Sign-MNIST dataset**.

It supports:
- Model training and evaluation  
- Real-time sign detection via webcam  
- Automatic graph generation and performance visualization  

##  Repository Structure
Sign-language-recognition
│
├── train_asl_model.py # Train & evaluate the CNN on Sign-MNIST dataset
├── asl_full_pipeline.py # Live ASL recognition using MediaPipe + trained CNN
├── graphy.py # Generate performance graphs & reports
│
├── improved_asl_model.h5 # Trained CNN model file
├── sign_mnist_test.csv # Test dataset (Sign-MNIST)
│
└── README.md # Project documentation


##  Features

 MediaPipe Integration – Real-time hand landmark tracking  
 Improved CNN Model – Optimized with batch normalization & dropout  
 Live Camera Recognition – Detects ASL letters in real time  
 Automatic Graphs – Accuracy, loss, confusion matrix & per-class reports  
 Data Augmentation – Enhances model robustness  
 User Calibration – Real-time ROI area feedback  

## 📊 Dataset

Dataset Used: [Sign Language MNIST (Kaggle)](https://www.kaggle.com/datamunge/sign-language-mnist)

| File | Description |
| `sign_mnist_train.csv` | 27,455 images of ASL letters (28×28 grayscale) |
| `sign_mnist_test.csv`  | 7,172 images for testing and validation |

Letters **J** and **Z** are excluded since they involve motion.

## 🧩 Model Architecture

Input: 28×28 grayscale hand gesture images  
Layers:  
  - Conv2D (32, 64, 128 filters)  
  - Batch Normalization  
  - MaxPooling2D + Dropout  
  - Dense (256 units → Softmax output for 25 classes)  
Optimizer:Adam  
Loss Function: Categorical Crossentropy  
Epochs: 25  
Batch Size: 128  

## 🚀 How to Run

### 1️⃣ Train the Model
python train_asl_model.py

This will:
Train the CNN model
Save it as improved_asl_model.h5
Display accuracy and loss plots

2️⃣ Generate Graphs
python graphy.py


Outputs visualizations:
Confusion matrix
Accuracy vs Epoch
Loss vs Epoch
Class distribution
Per-class accuracy
Sample predictions

All graphs are saved in the asl_graphs/ folder.

3️⃣ Run Real-Time ASL Detection
python asl_full_pipeline.py


Controls:
q → Quit
c → Calibrate area threshold

Live camera window shows:
Bounding box
Predicted letter
Confidence percentage
Detected area



4.Technologies Used

TensorFlow / Keras
MediaPipe   
OpenCV
NumPy / Pandas
Matplotlib / Seaborn
Scikit-learn







