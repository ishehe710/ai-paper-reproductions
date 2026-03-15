# LeNet-5 Architecture

## Overview

LeNet-5 is a convolutional neural network proposed by Yann LeCun et al. in 1998 for handwritten digit recognition. The network was originally designed to classify images from the MNIST dataset.

The architecture consists of alternating convolutional and pooling layers followed by fully connected layers. The original design uses **tanh activation functions**, **average pooling**, and a **Radial Basis Function (RBF) classifier** for the output layer.

This document describes the baseline architecture used for the reproduction.

---

## Input

* Image size: **32 × 32**
* Channels: **1 (grayscale)**

MNIST images (28 × 28) are resized to **32 × 32** to match the original LeNet-5 input size.

---

## Layer Architecture

### C1 – Convolution Layer

* Filters: 6
* Kernel size: 5 × 5
* Activation: tanh

Output size: **6 × 28 × 28**

---

### S2 – Subsampling (Average Pooling)

* Pooling type: Average pooling
* Kernel size: 2 × 2
* Stride: 2

Output size: **6 × 14 × 14**

---

### C3 – Convolution Layer

* Filters: 16
* Kernel size: 5 × 5
* Activation: tanh

Output size: **16 × 10 × 10**

---

### S4 – Subsampling (Average Pooling)

* Pooling type: Average pooling
* Kernel size: 2 × 2
* Stride: 2

Output size: **16 × 5 × 5**

---

### C5 – Convolution Layer

* Filters: 120
* Kernel size: 5 × 5
* Activation: tanh

Output size: **120 × 1 × 1**

---

### F6 – Fully Connected Layer

* Units: 84
* Activation: tanh

---

### Output Layer – RBF Classifier

The original LeNet-5 uses a **Radial Basis Function (RBF) classifier** instead of a softmax layer. Each output unit represents a class, and classification is performed based on the distance between the network output and learned class prototypes.

---

## Baseline Design Choices

The baseline reproduction follows the original LeNet-5 design:

* **Activation:** tanh
* **Pooling:** Average Pooling
* **Classifier:** Radial Basis Function (RBF)
* **Optimizer:** Stochastic Gradient Descent (SGD)

This baseline will later be compared with modern modifications such as ReLU activation, max pooling, softmax classification, and the Adam optimizer.
