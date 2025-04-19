# 🧑‍💻 Real-Time OCR of Digits 📸

A real-time Optical Character Recognition (OCR) system that extracts handwritten digits from live camera feeds using deep learning models trained on the MNIST dataset.

## 🚀 Project Overview

This project uses a Convolutional Neural Network (CNN) to recognize handwritten digits from images and live camera feeds. It leverages the MNIST dataset for training and uses OpenCV for capturing video input. The system is designed to provide real-time digit recognition for various applications such as automated data entry and digit tracking.

## 🧠 Features

- Real-time digit recognition from live video feeds.
- Uses a CNN model trained on the MNIST dataset.
- Captures video using OpenCV and processes frames for digit detection.
- Outputs recognized digits in real time on the display.

## 📊 Dataset

- **Source**: MNIST dataset of handwritten digits.
- **Attributes**: 28x28 grayscale images representing digits 0-9.

## 🛠️ Tech Stack

- Python
- TensorFlow/Keras (for deep learning model)
- OpenCV (for real-time video capture and image processing)
- NumPy, Matplotlib (for data manipulation and visualization)

## ⚙️ How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/sainipiyush05/Real-time-OCR-of-Digits.git
   cd Real-time-OCR-of-Digits
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the real-time OCR script:

bash
Copy
Edit
python real_time_ocr.py
The webcam will start, and the system will detect and display digits from the live feed.

📌 Screenshots
Add screenshots of the live OCR recognition if available.

✅ Model Accuracy
The model is trained on the MNIST dataset and achieves high accuracy in recognizing handwritten digits, providing reliable real-time performance.

📁 Folder Structure
sql
Copy
Edit
Real-time-OCR-of-Digits/
├── real_time_ocr.py
├── mnist_model.h5
├── requirements.txt
├── README.md
└── utils.py
👨‍💻 Author
Piyush Saini

🌟 Contribute
Feel free to fork the project and submit pull requests. Contributions and improvements are always welcome!

📄 License
This project is open-source and available under the MIT License.
