# Lung-Cancer-Detector
## Cancer Cell Detection & Classification using Hybrid Neural Network (CCDC-HNN)

Lung cancer is one of the most life-threatening diseases worldwide, primarily due to late-stage detection that limits treatment effectiveness. Early and accurate diagnosis significantly improves patient survival rates. Traditional diagnostic methods, such as manual CT scan evaluations and basic image processing algorithms, often lack precision and are time-consuming, hindering early-stage detection.

## Project Overview

This project presents a novel hybrid deep learning model, **CCDC-HNN**, designed to improve the efficiency and accuracy of lung cancer detection. It combines the strengths of:

- **Convolutional Neural Networks (CNN)** for effective feature extraction from lung CT scans,
- **Long Short-Term Memory (LSTM)** networks to capture temporal dependencies and changes in cancer cell structures over time,
- **UNET architecture** for precise image segmentation to isolate cancerous regions before classification.

The model is trained and tested using the publicly available **LIDC-IDRI dataset** of lung CT images. Pre-processing steps include image normalization and resizing to ensure consistent input.

Performance evaluation is based on metrics such as accuracy, precision, recall, and F1-score. The results demonstrate that CCDC-HNN reduces diagnosis time while achieving higher accuracy than existing techniques. It effectively distinguishes benign from malignant tumors, aiding medical professionals in treatment decisions and prognosis.

## Features

- Hybrid deep learning architecture combining CNN, LSTM, and UNET
- Precise segmentation of cancerous lung regions
- Robust classification of lung tumors
- User-friendly GUI built with Tkinter
- Implemented in Python using TensorFlow and Keras

## Tech Stack

- **Programming Language:** Python 3.x
- **Deep Learning Frameworks:** TensorFlow, Keras
- **Neural Network Architectures:**  
  - Convolutional Neural Network (CNN)  
  - Long Short-Term Memory (LSTM)  
  - UNET (for image segmentation)
- **Data Processing & Visualization:** NumPy, Pandas, Matplotlib, Seaborn, OpenCV
- **Dataset:** LIDC-IDRI (Lung Image Database Consortium image collection)
- **GUI:** Tkinter
- **Environment Management:** Virtualenv
- **Version Control:** Git & GitHub

## Evaluation

The performance of the CCDC-HNN model is assessed using the following metrics:

- **Accuracy:** Measures the overall correctness of the model in classifying lung cancer images.
- **Precision:** Indicates the proportion of true positive cancer detections among all positive predictions, minimizing false positives.
- **Recall (Sensitivity):** Reflects the model’s ability to correctly identify actual cancer cases, minimizing false negatives.
- **F1-Score:** The harmonic mean of precision and recall, providing a balanced measure of the model’s accuracy.
- **Segmentation Quality:** Assessed using metrics like Dice Coefficient or Intersection over Union (IoU) to evaluate the UNET’s ability to isolate cancerous regions accurately.

The model demonstrates improved diagnostic speed and higher accuracy compared to traditional techniques, effectively distinguishing between benign and malignant tumors.
