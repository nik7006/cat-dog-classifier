# 🐱🐶 Cat vs Dog Image Classification using CNN

## 📌 Project Overview

This project implements a Convolutional Neural Network (CNN) for binary image classification of cats and dogs. The model is trained using TensorFlow and Keras on a labeled dataset containing images of cats and dogs.

The objective is to automatically classify an input image as either a cat or a dog using deep learning techniques.

---

## 📂 Dataset Description

The dataset contains two classes:

- Cats
- Dogs

### Dataset Structure

```
training_set/
│
├── cats/
├── dogs/

test_set/
│
├── cats/
├── dogs/
```

### Image Specifications

- Image Size: 256 × 256
- Color Channels: RGB
- Classification Type: Binary Classification

---

## 🔍 Data Preprocessing

The following preprocessing steps were performed:

1. Image resizing to 256 × 256 pixels
2. Pixel normalization (0–255 → 0–1)
3. Batch loading using TensorFlow Dataset API
4. Training and validation dataset creation

```python
image = tf.cast(image / 255.0, tf.float32)
```

---

## 🧠 CNN Architecture

### Model Layers

| Layer | Configuration |
|---------|---------|
| Conv2D | 32 Filters (3×3) + ReLU |
| MaxPooling2D | 2×2 |
| Conv2D | 64 Filters (3×3) + ReLU |
| MaxPooling2D | 2×2 |
| Conv2D | 128 Filters (3×3) + ReLU |
| MaxPooling2D | 2×2 |
| Flatten | Feature Vector |
| Dense | 128 Neurons + ReLU |
| Dense | 64 Neurons + ReLU |
| Output | 1 Neuron + Sigmoid |

---

## ⚙️ Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- OpenCV
- Matplotlib
- Scikit-Learn

---

## 🚀 Model Training

### Training Configuration

| Parameter | Value |
|------------|---------|
| Optimizer | Adam |
| Loss Function | Binary Crossentropy |
| Batch Size | 32 |
| Epochs | 10 |
| Activation Function | ReLU / Sigmoid |

```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

---

## 📊 Performance Evaluation

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

### Results

| Metric | Score |
|---------|---------|
| Accuracy | 68.71% |
| Precision | 66.64% |
| Recall | 75.00% |
| F1-Score | 70.57% |

---

## 📈 Classification Report

| Class | Precision | Recall | F1-Score |
|---------|---------|---------|---------|
| Cat | 0.71 | 0.62 | 0.67 |
| Dog | 0.67 | 0.75 | 0.71 |

---

## 🔲 Confusion Matrix

```
[[631 380]
 [253 759]]
```

Interpretation:

- Correctly Classified Cats: 631
- Misclassified Cats: 380
- Correctly Classified Dogs: 759
- Misclassified Dogs: 253

---

## 🖼️ Sample Prediction

The trained model can predict whether a new image contains a cat or a dog.

```python
prediction = model.predict(test_input)

if prediction < 0.5:
    print("CAT")
else:
    print("DOG")
```

---

## 💾 Model Saving

```python
model.save('cat_dog_classifier.h5')
```

---

## 📌 Future Improvements

- Data Augmentation
- Transfer Learning (MobileNetV2, ResNet50)
- Hyperparameter Tuning
- Increased Training Epochs
- Improved CNN Architecture

---

## 📷 Output Examples

### Training Accuracy Graph

(Add Screenshot)

### Training Loss Graph

(Add Screenshot)

### Confusion Matrix

(Add Screenshot)

---
