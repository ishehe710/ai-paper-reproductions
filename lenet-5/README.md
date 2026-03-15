# LeNet-5 Reproduction (PyTorch)

## Overview

This project reproduces the classic **LeNet-5 convolutional neural network**, introduced by Yann LeCun in the paper [**Gradient-Based Learning Applied to Document Recognition (1998)**](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf).

LeNet-5 was one of the first successful convolutional neural networks and played a key role in demonstrating the effectiveness of gradient-based learning for image recognition tasks.

In this project, the architecture is reproduced using the modern deep learning framework **PyTorch**. Several experiments are conducted to evaluate how the original design choices compare to commonly used modern deep learning components.

Dataset used: **MNIST** handwritten digits.

---

## Architecture

The original LeNet-5 architecture consists of alternating convolution and subsampling layers followed by fully connected layers.

Input: 1×28×28 grayscale digit image

```
Input (1 × 28 × 28)
↓
C1: Conv(1 → 6, 5×5) + Scaled Tanh
↓
S2: AvgPool(2×2)
↓
C3: Conv(6 → 16, 5×5) + Scaled Tanh
↓
S4: AvgPool(2×2)
↓
C5: Conv(16 → 120, 5×5)
↓
F6: Fully Connected (120 → 84)
↓
Output Layer (RBF Classifier)
```

The original model used **tanh activation function** and an **RBF output layer** for classification, which was designed specifically for the architecture.

---

## Experiments

Several variations of the model were implemented to compare historical and modern design choices.

| Model                           | Activation  | Pooling | Optimizer | Classifier       | BatchNorm | Training Time | Test Accuracy | Notes                                         |
| ------------------------------- | ----------- | ------- | --------- | ---------------- | --------- | ------------- | ------------- | --------------------------------------------- |
| **Baseline LeNet-5**            | Scaled Tanh | AvgPool | SGD       | RBF              | No        | 3m 37s        | **95.78%**    | Reproduction of original LeNet-5 architecture |
| **ReLU**                        | ReLU        | AvgPool | SGD       | RBF              | No        | 3m 22s        | 72.12%        | Activation mismatch with RBF classifier       |
| **MaxPool**                     | Scaled Tanh | MaxPool | SGD       | RBF              | No        | 3m 58s        | 94.01%        | Replace AvgPool with MaxPool                  |
| **Adam**                        | Scaled Tanh | AvgPool | Adam      | RBF              | No        | 3m 55s        | 96.42%        | Replace SGD with Adam optimizer               |
| **Softmax Classifier**          | Scaled Tanh | AvgPool | SGD       | Linear + Softmax | No        | 4m 14s        | 93.82%        | Learnable fully connected classifier          |
| **BatchNorm**                   | Scaled Tanh | AvgPool | SGD       | RBF              | Yes       | 4m 31s        | 96.41%        | Batch normalization after convolution layers  |
| **ReLU + Learnable Classifier** | ReLU        | AvgPool | SGD       | Linear + Softmax | No        | 3m 45s        | 92.32%        | ReLU paired with Softmax classifier           |
| **Modern LeNet**                | ReLU        | MaxPool | Adam      | Linear + Softmax | Yes       | 4m 04s        | **99.27%**    | Combined modern improvements                  |


## Implementation Notes

While the original LeNet-5 architecture used several design choices specific to the hardware and training methods available in the 1990s, this reproduction adapts the architecture to modern deep learning workflows while preserving the core structure.

Key implementation details:

- The original **scaled tanh activation** was implemented to match the behavior described in the paper.
- The **RBF output layer** was reproduced to approximate the original classifier design.
- Experiments compare these historical components with modern alternatives such as **ReLU activations**, **Batch Normalization**, and **Softmax classifiers**.

---

### Key Observation

ReLU does not pair well with the original RBF output layer because:

* ReLU range: `[0, ∞)`
* tanh range: `[-1, 1]`

The original RBF layer was scaled to match the tanh activation range.

Therefore a modern variant using **ReLU + Softmax** was implemented to produce stable training.

---

## Results

The experiments demonstrate that while the original architecture works well, modern activation and output functions can improve training stability and performance when implemented in contemporary frameworks.

---

## Key Learnings

Through this reproduction the following concepts were explored:

* Early convolutional neural network design
* Historical activation functions
* Output layer design (RBF vs Softmax)
* CNN training using PyTorch

---

## Technologies Used

* PyTorch
* Python
* NumPy
* Matplotlib

---

## Project Structure

States the relevant files of the project
```
lenet5
│
├── docs
   ├── architecture.md
   └── paper_summary.md  
├── models
   └── lenet5.py  
├── notebooks 
   └── lenet_5_experiments.ipynb   
├── notes
└── README.md
```
