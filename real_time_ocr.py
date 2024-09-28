import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
import os

# Check if the model file exists
model_path = 'mnist_cnn_model.h5'
if not os.path.exists(model_path):
    print(f"Error: Model file '{model_path}' not found.")
    print("Please run the training script first to create the model.")
    exit(1)

# Load the pre-trained digit recognition model
try:
    model = load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading the model: {str(e)}")
    exit(1)

# Initialize video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit(1)

# Performance metrics
frame_count = 0
start_time = time.time()
total_inference_time = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Pre-process the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            roi = gray[y:y+h, x:x+w]
            
            # Resize and normalize the ROI
            roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
            roi = roi.astype('float32') / 255.0
            roi = np.expand_dims(roi, axis=-1)
            roi = np.expand_dims(roi, axis=0)

            # Perform inference
            inference_start = time.time()
            prediction = model.predict(roi, verbose=0)
            inference_time = time.time() - inference_start
            total_inference_time += inference_time

            digit = np.argmax(prediction)
            confidence = np.max(prediction) * 100

            # Draw bounding box and text
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Digit: {digit} ({confidence:.2f}%)", 
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Calculate and display FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Real-time Digit OCR', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Calculate average inference time
avg_inference_time = total_inference_time / frame_count if frame_count > 0 else 0

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()

# Print performance metrics
print(f"Average FPS: {fps:.2f}")
print(f"Average inference time: {avg_inference_time*1000:.2f} ms")