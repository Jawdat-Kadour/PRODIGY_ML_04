
# Hand Gesture Recognition Using CNN
This project implements a real-time hand gesture recognition system using a Convolutional Neural Network (CNN). The model is trained on a dataset of hand gestures and can classify different gestures from live video or images. The system uses OpenCV for image processing and Keras with TensorFlow backend for deep learning.

### Introduction
This project aims to recognize and classify hand gestures using a deep learning model. Hand gestures are a key component in human-computer interaction, enabling touch-free control of devices and improving accessibility. The project leverages CNN to classify different gestures and can be used in real-time applications, including gesture-based control systems and augmented/virtual reality.

### Dataset
The dataset contains images of 10 different hand gestures (e.g., palm, thumb, fist, etc.) performed by various subjects. Each gesture is stored in a separate folder, and the images are grayscale.

Gesture Classes: Palm, L, Fist, Fist Moved, Thumb, Index, Ok, Palm Moved, C, Down
Data Size: 20,000 images (240x640 resolution)

link: https://www.kaggle.com/datasets/gti-upm/leapgestrecog

### Model Architecture
The model is built using Keras with TensorFlow backend. It includes the following layers:

Convolutional Layers: Three Conv2D layers with ReLU activation to capture spatial features.
Pooling Layers: MaxPooling2D layers to reduce dimensionality.
Fully Connected Layer: A Dense layer with 64 units and ReLU activation.
Dropout Layer: Added to prevent overfitting during training.
Output Layer: A Dense layer with 10 units (corresponding to 10 gestures) and softmax activation.
Preprocessing
Grayscale Conversion: All images are converted to grayscale.
Resizing: Images are resized to a smaller resolution (120x320) to reduce the computational load.
Normalization: Pixel values are normalized between 0 and 1 by dividing by 255.
Reshaping: The data is reshaped to include a single channel for grayscale images.

### Training
Optimizer: Adam
Loss Function: Sparse categorical crossentropy
Metrics: Accuracy
The model was trained for 10 epochs with a batch size of 32.
bash
Copy code
500/500 [==============================] - 101s 201ms/step - loss: 0.0109 - accuracy: 0.9961 - val_loss: 7.4536e-06 - val_accuracy: 1.0000

### Evaluation
The model was evaluated on the test set, and it achieved a 100% accuracy.

bash
Copy code
125/125 [==============================] - 4s 34ms/step - loss: 7.4536e-06 - accuracy: 1.0000
Test Accuracy: 100.00%

### Real-Time Testing
The project includes a real-time hand gesture recognition system using OpenCV. The system captures frames from a live video feed, processes them, and classifies the gesture in real time.

Example Usage
Open the webcam feed.
Preprocess the frame (convert to grayscale, resize, normalize).
Pass the frame through the trained model for prediction.
Display the predicted gesture on the video feed.

### Installation
Clone the repository:

bash
Copy code
git clone https://github.com/Jawdat-Kadour/PRODIGY_ML_04.git
Navigate to the project directory:

bash
Copy code
cd hand-gesture-recognition
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt

### Usage
Train the model (if you want to retrain it):

bash
Copy code
python train_model.py
Test the model with a live video feed:

bash
Copy code
python real_time_gesture_recognition.py

### Results
The model was able to accurately classify hand gestures with near-perfect accuracy on both the training and validation sets. While testing with live video, the system was able to detect gestures such as thumbs up, fist, and palm accurately in real-time.