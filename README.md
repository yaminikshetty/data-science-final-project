📁 Project Structure


ASL-Recognition-Ultra/
│
├── train_asl_model.py         # Trains the CNN using Sign-MNIST dataset
├── asl_full_pipeline.py       # Real-time ASL recognition via webcam + MediaPipe
├── graphy.py                  # Generates model performance graphs
│
├── improved_asl_model.h5      # Trained CNN model file (auto-generated)
├── sign_mnist_train.csv       # Training dataset
├── sign_mnist_test.csv        # Testing dataset
│
└── README.md                  # Documentation file


⚙️ Requirements & Dependencies

🧩 Required Software

Python 3.8+
pip (Python package manager)
Webcam (for live ASL detection)

📦 Python Dependencies

Install all dependencies using the following command:

pip install -r requirements.txt


#If you don’t have a requirements.txt, create one with this content:

tensorflow==2.15.0
mediapipe==0.10.9
opencv-python
numpy
pandas
matplotlib
seaborn
scikit-learn


💡 Tip: Use a virtual environment to avoid dependency conflicts:

-python -m venv venv
-source venv/bin/activate  # on Windows: venv\Scripts\activate


🧠 Dataset Details

Dataset: Sign Language MNIST

File	Description
sign_mnist_train.csv	Training images (27,455 samples, 28×28 grayscale)
sign_mnist_test.csv	Test images (7,172 samples, 28×28 grayscale)

Note: Letters J and Z are excluded as they involve motion gestures.

🧩 Model Architecture

The CNN model used for classification includes:
Conv2D Layers: (32, 64, 128 filters)
Batch Normalization
MaxPooling + Dropout Layers
Flatten + Dense Layers
Activation: ReLU and Softmax
Optimizer: Adam
Loss Function: Categorical Crossentropy
Output Classes: 25 (A–Y)

🚀 How to Run the Project

Step 1️⃣ – Train the Model

#To train your CNN model using the Sign-MNIST dataset:

python train_asl_model.py


This will:

-Load and preprocess data
-Train the CNN model
-Display accuracy/loss graphs
-Save the model as improved_asl_model.h5

🗂 Output:

improved_asl_model.h5 (trained model)


Step 2️⃣ – Generate Performance Graphs

To analyze the trained model and visualize metrics:

python graphy.py


This script generates:
-Confusion Matrix
-Accuracy vs. Epochs
-Loss vs. Epochs
-Per-Class Accuracy
-Class Distribution

Random Sample Predictions
📂 All graphs are saved in the folder:

asl_graphs/

Step 3️⃣ – Run Real-Time ASL Detection

To start live camera detection using MediaPipe and the trained CNN:

python asl_full_pipeline.py


Controls during runtime:

Key	Action
q	Quit live window
c	Print area calibration value

🎥 Output Display Includes:

-Detected ASL letter
-Confidence percentage
-ROI bounding box

Hand area feedback

📊 Output Examples

Visualization	                                                 Description
Confusion Matrix	                             Shows classification accuracy for each letter
Accuracy/Loss Plot	                                   Displays model learning curve
Per-Class Accuracy	                              Highlights accuracy variation by letter
Sample Predictions	                              Shows actual vs. predicted ASL gestures
Live Detection	                                Webcam-based real-time gesture recognition
🔧 Configuration                                 Variables (Inside asl_full_pipeline.py)


Variable	                                                      Description	                                                        Default
MODEL_PATH	                                                 Path to trained model                                       	"improved_asl_model.h5"
TRAIN_CSV, TEST_CSV	                                            Dataset paths//                                  "sign_mnist_train.csv", "sign_mnist_test.csv"
IMG_SIZE                                                  	Image resize for CNN	                                                    28
CONF_THRESH	                                      Confidence threshold for stable prediction	                                       0.75
STABLE_FRAMES                              	Number of consistent frames before confirming a letter                                     	8
COOLDOWN_FRAMES                                      	Delay before next prediction	                                                10
MIN_BBOX_AREA	                                     Minimum ROI size to validate hand                                              	800

You can tweak these settings for higher stability or responsiveness.

📁 Supporting Files


File	                                                            Purpose
train_asl_model.py	                              Builds, trains, and evaluates CNN model
asl_full_pipeline.py	                      Real-time MediaPipe + CNN recognition pipeline
graphy.py	                                    Generates analysis and visualization graphs
sign_mnist_train.csv, sign_mnist_test.csv	                      Datasets
improved_asl_model.h5                                     Trained CNN model file

💡 Future Enhancements

-Add J & Z motion tracking using temporal data
-Convert to TensorFlow Lite for mobile deployment
-Integrate voice/text output for recognized gestures
-Build web interface using TensorFlow.js
