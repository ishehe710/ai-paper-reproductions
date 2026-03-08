# LeNet-5 Reproduction and Modern Improvements

## Project Overview

This project reproduces the original **LeNet-5 convolutional neural network** architecture proposed by Yann LeCun et al. The goal is to implement the model as closely as possible to the original paper using PyTorch, and then explore how modern deep learning improvements affect its performance.

The baseline model follows the design choices described in the original paper, including:

* Hyperbolic tangent (tanh) activation functions
* Average pooling layers
* Radial Basis Function (RBF) classifier

After reproducing the original model, several experiments will introduce modern improvements to analyze why these techniques became standard in modern neural networks.

---

## Requirements

* Python 3.10+
* PyTorch
* torchvision
* numpy
* matplotlib

Install dependencies with:

pip install torch torchvision numpy matplotlib

---

## Project Structure

project/
│
├── models/        # Model implementations
├── experiments/   # Experimental modifications
├── train.py       # Training script
├── evaluate.py    # Evaluation script
└── README.md

---

## Running the Baseline Model

To train the baseline LeNet-5 implementation:

python train.py

This will train the model using the original architectural choices described in the LeNet-5 paper.

---

## Planned Experiments

The following experiments will evaluate common modern improvements to the original architecture.

1. **Activation Function**

   * Replace **tanh** with **ReLU**

2. **Classifier**

   * Replace **RBF classifier** with **Softmax**

3. **Pooling Strategy**

   * Replace **Average Pooling** with **Max Pooling**

4. **Optimizer**

   * Compare **SGD** with **Adam**

5. **Normalization**

   * Add **Batch Normalization**

These experiments will help analyze how modern techniques influence training stability, convergence speed, and overall performance.

---

## Goal

The objective of this project is to understand the design choices behind early convolutional neural networks and evaluate why modern improvements such as ReLU, Batch Normalization, and Adam optimization became widely adopted.
