# 🐱🐶 Cat vs Dog Image Classification using CNN

## 📌 Project Overview

This project implements a Convolutional Neural Network (CNN) for binary image classification of cats and dogs. The model is trained using TensorFlow and Keras on a labeled dataset containing images of cats and dogs.

The objective is to automatically classify an input image as either a cat or a dog using deep learning techniques.

---
##  Architecture

<img width="1742" height="764" alt="image" src="https://github.com/user-attachments/assets/fd6a8248-1d9d-43e1-8372-1fe42774c90c" />



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
| Accuracy | 75.28% |
| Precision | 76% |
| Recall | 76% |
| F1-Score | 76% |

---

## 📈 Classification Report

| Class | Precision | Recall | F1-Score |
|---------|---------|---------|---------|
| Cat | 0.76 | 0.75 | 0.76 |
| Dog | 0.76 | 0.76 | 0.76 |
---

<img width="350" height="200" alt="image" src="https://github.com/user-attachments/assets/ee1e0e2d-7da7-41c5-9766-18815d512706" />
---
Overall Accuracy

**75.28%**


## 🔲 Confusion Matrix

```text
[[761 250]
 [240 772]]
```

### Interpretation

- Correctly Classified Cats = 761
- Incorrectly Classified Cats = 250
- Correctly Classified Dogs = 772
- Incorrectly Classified Dogs = 240

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

<img width="864" height="790" alt="image" src="https://github.com/user-attachments/assets/bb608ea9-4597-4f4f-8929-39cd676ba93e" />

### Training Loss Graph

<img width="600" height="914" alt="image" src="https://github.com/user-attachments/assets/d726493e-e5d4-4939-bae8-5d7610434be7" />

### Confusion Matrix

<img width="1030" height="938" alt="image" src="https://github.com/user-attachments/assets/d3f04dee-5ca6-4284-98d5-0adc6a71eb82" />

---
## Conclusion
This project successfully developed a Convolutional Neural Network (CNN) for the classification of cat and dog images. The model was trained and tested on a labeled dataset containing images from two categories: cats and dogs.

The CNN was able to learn important visual features from the input images and classify them with an overall accuracy of 75.28%. The model achieved a Precision of 76%, Recall of 76%, and an F1-Score of 76%, indicating reliable and balanced performance across both classes.

The confusion matrix and classification report demonstrate that the model effectively distinguishes between cat and dog images while maintaining consistent prediction accuracy. The results confirm that Convolutional Neural Networks are well-suited for image classification tasks and can successfully extract meaningful patterns from visual data.

Overall, this project provides practical experience in deep learning, image preprocessing, model training, performance evaluation, and computer vision applications. The developed classifier serves as a solid foundation for more advanced image recognition systems and real-world deep learning applications

