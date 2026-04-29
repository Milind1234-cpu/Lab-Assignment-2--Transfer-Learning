# Lab Assignment 2 - Transfer Learning for Pneumonia Detection

## 📋 Project Overview

This project implements a deep learning model using **Transfer Learning** with the **VGG16** pre-trained model to classify chest X-ray images for pneumonia detection. This work is part of a Deep Learning course lab assignment focused on research paper implementation using pre-trained models.

## 👥 Team Information

- **Student Name:** Milind Lanje
- **Student ID:** 202402100010
- **Course:** Deep Learning
- **Date of Submission:** April 26, 2024
- **Group Members:** Yash Chopade, Abhishek Lipne, Sakib Attar

## 🎯 Objectives

1. Study a research paper utilizing a pre-trained model
2. Reproduce the model implementation using the dataset and methodology from the research paper
3. Fine-tune the pre-trained model and optimize hyperparameters
4. Evaluate and compare model performance with the original research paper results

## 📊 Dataset

**Dataset:** [Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

The dataset contains chest X-ray images organized into three splits:
- **Training Set:** 5,216 images (1,341 Normal, 3,875 Pneumonia)
- **Validation Set:** 16 images (8 Normal, 8 Pneumonia)
- **Test Set:** 624 images (234 Normal, 390 Pneumonia)

### Classes
- **NORMAL:** Healthy chest X-rays
- **PNEUMONIA:** Chest X-rays showing pneumonia

## 🏗️ Model Architecture

### Pre-trained Model: VGG16
- **Base Model:** VGG16 (pre-trained on ImageNet)
- **Input Size:** 224x224 pixels
- **Transfer Learning Approach:** Feature extraction with fine-tuning

### Custom Layers
- Global Average Pooling 2D
- Batch Normalization
- Dense layers with Dropout
- Binary classification output

## 🔧 Implementation Details

### Data Preprocessing
- **Image Resizing:** 224x224 pixels (VGG16 input size)
- **Normalization:** Pixel values scaled to [0, 1]
- **Data Augmentation (Training only):**
  - Rotation: ±10 degrees
  - Width/Height shift: 10%
  - Shear transformation: 10%
  - Zoom: 10%
  - Horizontal flip

### Training Configuration
- **Batch Size:** 32
- **Optimizer:** Adam
- **Callbacks:**
  - Early Stopping
  - Learning Rate Reduction on Plateau
  - Model Checkpoint

## 📦 Requirements

```python
tensorflow >= 2.19.0
numpy
matplotlib
seaborn
pandas
scikit-learn
kaggle
```

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Milind1234-cpu/Lab-Assignment-2--Transfer-Learning.git
cd Lab-Assignment-2--Transfer-Learning
```

### 2. Install Dependencies
```bash
pip install tensorflow numpy matplotlib seaborn pandas scikit-learn kaggle
```

### 3. Download Dataset
The notebook includes code to automatically download the dataset from Kaggle. You'll need to:
1. Create a Kaggle account
2. Generate an API token
3. Configure the Kaggle API credentials in the notebook

### 4. Run the Notebook
Open `Lab_Assignment_2_Submission (1).ipynb` in Google Colab or Jupyter Notebook and run all cells.

## 📈 Evaluation Metrics

The model is evaluated using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **Confusion Matrix**
- **Classification Report**

## 🔍 Key Features

✅ Transfer Learning with VGG16  
✅ Data Augmentation for improved generalization  
✅ Binary classification (Normal vs Pneumonia)  
✅ Comprehensive evaluation metrics  
✅ Visualization of training history  
✅ GPU acceleration support  

## 📝 Tasks Completed

### Task 1: Research Paper Selection and Dataset Preparation
- ✅ Selected research paper on pre-trained models
- ✅ Identified and downloaded the Chest X-Ray Pneumonia dataset
- ✅ Performed preprocessing (resizing, normalization)
- ✅ Applied data augmentation techniques
- ✅ Split dataset into training, validation, and testing sets

### Task 2: Model Implementation
- ✅ Implemented VGG16 transfer learning architecture
- ✅ Fine-tuned the model with custom layers
- ✅ Optimized hyperparameters

### Task 3: Evaluation
- ✅ Evaluated model performance on test set
- ✅ Generated classification metrics
- ✅ Compared results with research paper findings

## 🖥️ Hardware Requirements

- **Recommended:** GPU-enabled environment (Google Colab with GPU runtime)
- **Minimum:** CPU with 8GB RAM (training will be slower)

## 📄 License

This project is for educational purposes as part of a Deep Learning course assignment.

## 🙏 Acknowledgments

- Dataset: [Paul Mooney - Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- Pre-trained Model: VGG16 from Keras Applications
- Framework: TensorFlow/Keras

## 📧 Contact

For questions or collaboration:
- **Email:** milindlanje125@gmail.com
- **GitHub:** [Milind1234-cpu](https://github.com/Milind1234-cpu)

---

**Note:** This implementation is part of an academic assignment and follows the methodology described in the selected research paper on transfer learning for medical image classification.
