# Brain Tumor Classification and Explainable Analysis

## Overview

This repository contains the experimental code and resources used in my machine learning project:

**"Brain Tumor Classification from MRI Images Using a Custom CNN, XceptionNet, EfficientNet-B0, and Vision Transformer with Explainable AI Analysis"**

Author: **Yujin Jeon**

The main objective of this study is to evaluate and compare several state-of-the-art deep learning models for brain tumor classification on MRI images, and to analyze their decision-making processes using explainable AI (XAI) techniques.

---

## Objectives

- Evaluate and compare the performance of four models:
  - Custom CNN
  - XceptionNet
  - EfficientNet-B0
  - Vision Transformer (ViT)

- Investigate model explainability using Grad-CAM.

- Analyze the focus areas and interpretability on MRI images for different tumor types.

---

## Models and Techniques

- **Custom CNN**: A basic convolutional neural network built from scratch to serve as a baseline.
- **XceptionNet**: Depthwise separable convolutions, effective for medical image analysis.
- **EfficientNet-B0**: Highly efficient and scalable convolutional architecture.
- **ViT (Vision Transformer)**: Transformer-based architecture capturing global features effectively.

### Explainable AI (XAI)

- **Grad-CAM**: Applied to CNN-based models (Custom CNN, XceptionNet, EfficientNet-B0) for visualizing important activation regions.

---

## Dataset

- Brain tumor MRI image dataset prepared and curated for this study.
- Dataset details, preprocessing pipeline, and class distribution are described in the report.

---

## Code

The full experimental code is included in this repository as a Jupyter Notebook.  
Please refer to the notebook for detailed implementation and experiments.

---

## Results

Detailed results, performance metrics, and visual analysis are described in the report.

---

## Report

The report is included:

- [/Paper/Report_Brain_Tumor_Classification_Yujin_Jeon.pdf](/Paper/Report_Brain_Tumor_Classification_Yujin_Jeon.pdf)

---

## Repository Structure

- `/Code/Code_Brain_Tumor_Classification_Yujin_Jeon.ipynb`: Jupyter Notebook containing all code versions, illustrating step-by-step research progress.
- `/Paper/Report_Brain_Tumor_Classification_Yujin_Jeon.pdf`: Report.
- `models/`: Saved model metrics.

---

## Frameworks and Libraries Used

### Main deep learning and model libraries
- PyTorch
- timm
- torchvision
- EfficientNet-PyTorch
- Transformers (Hugging Face)

### Explainable AI (XAI)
- Grad-CAM (pytorch-grad-cam)

### Evaluation and analysis
- scikit-learn

### Image processing
- PIL (Python Imaging Library)

### Visualization
- Matplotlib
- Seaborn

### Utilities and others
- Google Colab
- NumPy
- Pandas
- Joblib

---

## Contact

For any questions, feel free to contact:

- **Yujin Jeon**
- Email: yujin.jeon.developer@gmail.com
