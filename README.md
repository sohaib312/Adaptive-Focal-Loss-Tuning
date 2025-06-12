# Adaptive Focal Loss with Bayesian Optimization on CIFAR-10

This project implements a CNN-based image classification pipeline on the CIFAR-10 dataset using **Adaptive Focal Loss (AFL)**. It integrates **Bayesian Optimization (via scikit-optimize)** to fine-tune the `gamma` and `alpha` hyperparameters of the AFL loss function, aiming to improve model performance on imbalanced or difficult samples.

---

## 🔍 Motivation

Traditional loss functions like cross-entropy do not perform well when the dataset has class imbalance or hard-to-classify samples. Focal Loss helps by down-weighting easy samples and focusing more on hard ones. This project extends it with **adaptive tuning** of focal parameters using Bayesian Optimization.

---

## 🚀 Features

- CNN model built with **TensorFlow/Keras**
- Custom **Adaptive Focal Loss** implementation
- Hyperparameter tuning of `gamma` and `alpha` using **Bayesian Optimization**
- Evaluation using **Stratified K-Fold Cross Validation**
- Comparison of default AFL and tuned AFL on CIFAR-10

---

## 🧠 Model Architecture

- Conv2D (32 filters) → MaxPooling
- Conv2D (64 filters) → MaxPooling
- Flatten → Dense(128) → Dense(10 with Softmax)

---

## 🧪 Results

- ✅ **Default AFL Accuracy** on test set  
- ✅ **Tuned AFL Accuracy** after 5-fold cross-validation  
- 🔍 Search space: `gamma ∈ [1.0, 5.0]`, `alpha ∈ [0.1, 0.5]`

---

## 📊 Dependencies

Make sure to install the following:

```bash
pip install tensorflow scikit-optimize numpy matplotlib scikit-learn
