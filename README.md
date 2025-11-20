# 🍱 Food Image Classification – Deep Learning  
**Assignment 1 • Individual Project (AY2025)**  
**Author:** Tan Xin Yi

This project is an end-to-end deep learning pipeline that classifies images of **10 different food categories**.  
Using Python, TensorFlow, and Convolutional Neural Networks (CNNs), the model learns to recognize visual patterns in images and predict the correct food class with strong accuracy.

This assignment demonstrates my skills in **computer vision**, **data preprocessing**, **CNN model design**, and **model evaluation**.

---

## 📘 Project Overview

Modern dietary tracking apps and smart kitchen systems rely on accurate food recognition.  
This project simulates a real-world classification challenge by training a model that can:

- Load and preprocess raw images  
- Augment data to improve generalization  
- Train a CNN using TensorFlow/Keras  
- Evaluate performance using validation/testing sets  
- Classify new food images into one of 10 classes  

The final trained model can be used to recognize food images instantly.

---

## 🧂 Dataset

- Source: **Kaggle Food-101 (subset)**  
- 10 food classes selected  
- Image size standardized to **150×150**  
- Dataset split into:
  - **Training set**
  - **Validation set**
  - **Testing set**

A fixed seed (`42`) ensures reproducibility.

---

## 🧹 Step 1 — Data Loading & Preprocessing

Key preprocessing steps:

- Resize all images to **150×150 pixels**
- Rescale pixel values from `0–255` to `0–1`
- Set batch size to **32**
- Use three data generators:
  - **Training generator** (with augmentation)
  - **Validation generator**
  - **Test generator**

### ✔ Data Augmentation Techniques

- Rotation  
- Zoom  
- Width & height shift  
- Horizontal flip  

These help prevent overfitting and allow the model to generalize better.

---

## 🧠 Step 2 — Model Building

A custom **Convolutional Neural Network (CNN)** was built using TensorFlow/Keras.

### **Model Architecture Overview**

- Convolutional layers with ReLU activation  
- MaxPool layers to reduce spatial dimensions  
- Dropout layers to reduce overfitting  
- Flatten + Dense layers  
- Final Softmax layer for 10-class output  

### **Compilation**

- Loss: `categorical_crossentropy`  
- Optimizer: `Adam`  
- Metrics: `accuracy`  

---

## 🏋️ Step 3 — Model Training

The model was trained using:

- **Training data** with augmentation  
- **Validation data** for performance tracking  
- Early stopping to avoid overfitting

### Training results include:

- Training Loss & Accuracy plots  
- Validation Loss & Accuracy plots  

These visualizations help interpret the model’s learning behaviour.

---

## 🧪 Step 4 — Model Evaluation

Model performance was evaluated using:

- Test accuracy  
- Loss scores  
- Correct vs incorrect predictions  
- Ability to generalize to new images

The final model demonstrated strong classification performance across all 10 food categories.

---

## 🍽️ Step 5 — Prediction on New Images

The completed model can classify **any external food image**.

Example pipeline:
Input Image → Preprocess → Model → Predicted Food Class


Output includes:

- Predicted class name / Food Name
- Confidence probability  
- Visualization of the input and predicted result  

---

## 📂 Project Structure
Assignment1/
│── ASG_Problem1_Tan_Xin_Yi.ipynb
│── README.md


---

## 🛠️ Technologies Used

- **Python**
- **TensorFlow / Keras**
- **NumPy & Pandas**
- **Matplotlib**
- **scikit-learn**
- **Google Colab / Jupyter Notebook**

---

## 🎯 Key Skills Demonstrated

- Deep learning model design  
- CNN architecture building  
- Image preprocessing & augmentation  
- Model evaluation & visualization  
- Reproducible experimentation  
- Applying DL to real-world food classification  

---


## 📌 Summary

This project is a complete deep learning workflow built from scratch — covering image preparation, CNN modeling, training, evaluation, and prediction.  
It demonstrates practical competence in **computer vision**, **TensorFlow**, and **deep learning model development**.



