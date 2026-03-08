# Paper Summary: LeNet-5

**Paper:** Gradient-Based Learning Applied to Document Recognition

**Authors:** Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick Haffner

**Year:** 1998

## Problem

Before convolutional neural networks became widely used, many character recognition systems relied on manually designed features. These handcrafted features required domain expertise and often struggled to generalize well to variations in handwriting.

The goal of this work is to develop a neural network architecture capable of automatically learning useful features directly from raw pixel inputs. By learning hierarchical representations of visual data, the system can perform handwritten digit recognition more robustly without requiring manual feature engineering.

## Key Ideas

The convolutional neural network proposed in this paper combines several architectural ideas that improve robustness to translation, distortion, and small variations in the input images.

First, the network uses **local receptive fields**, meaning each neuron only processes information from a small region of the input image. This allows the network to detect local visual patterns such as edges or strokes.

Second, the architecture uses **shared weights** within feature maps. Units within a feature map apply the same learned filter to different spatial locations of the input image. This weight sharing significantly reduces the number of trainable parameters while allowing the network to detect the same feature across different parts of the image.

Third, the network includes **subsampling layers** that perform local averaging and reduce the spatial resolution of feature maps. These layers decrease the sensitivity of the network to small shifts or distortions in the input.

Together, these design principles allow the network to learn hierarchical features while maintaining robustness to variations in the input data.

## Architecture

The LeNet-5 architecture consists of seven layers with trainable parameters (excluding the input layer). The network processes normalized grayscale images of size **32 × 32 pixels**.

The architecture follows this general structure:

Input: 32 × 32 grayscale image

C1 – Convolutional layer with 6 feature maps of size 28 × 28. Each unit is connected to a 5 × 5 neighborhood in the input. This layer contains 156 trainable parameters.

S2 – Subsampling layer with 6 feature maps of size 14 × 14. Each unit is connected to a 2 × 2 region in the corresponding C1 feature map.

C3 – Convolutional layer with 16 feature maps. Units are connected to several 5 × 5 neighborhoods in subsets of the S2 feature maps. This layer contains 1,516 trainable parameters.

S4 – Subsampling layer with 16 feature maps of size 5 × 5.

C5 – Convolutional layer with 120 feature maps. Each unit is connected to a 5 × 5 neighborhood covering all 16 feature maps from S4.

F6 – Fully connected layer with 84 units connected to the outputs of C5.

Output – The output layer contains one unit for each class.

## Activation Function

Units in layers up to F6 compute a weighted sum of their inputs plus a bias term:

y = wᵀx + b

This value is then passed through a **scaled hyperbolic tangent activation function**:

f(a) = A tanh(Sa)

where **A = 1.7159** and **S controls the slope of the function near the origin**.

## Output Layer and Loss Function

The original LeNet-5 architecture uses **Euclidean Radial Basis Function (RBF) units** in the output layer. Each RBF unit computes the distance between the input vector and a learned parameter vector representing a class prototype.

The output of each unit is computed as:

yᵢ = − Σⱼ (xⱼ − wᵢⱼ)²

This allows the network to measure how close the input representation is to each class prototype.

The network is trained using a **Maximum Likelihood Estimation (MLE) criterion**. The loss function encourages the network to reduce the error for the correct class while increasing the error for incorrect classes. Training is performed using **backpropagation** to compute gradients for all weights in the convolutional network.

## Significance

LeNet-5 demonstrated that convolutional neural networks could successfully learn hierarchical visual features directly from pixel data. By combining convolutional layers, shared weights, and subsampling operations, the architecture achieved robust handwritten digit recognition.

Many of the core ideas introduced in this architecture—including convolutional feature extraction, weight sharing, and pooling—remain fundamental components of modern deep learning systems.

## Notes for Reproduction

To reproduce the LeNet-5 model, I will implement the architecture using PyTorch while attempting to follow the procedure described in the original paper as closely as possible. This includes using hyperbolic tangent (tanh) as the activation function and radial basis function (RBF) units for the final classification layer, as described by the authors.

This implementation will serve as the baseline model. After reproducing the original architecture, I will conduct experiments that introduce modern improvements commonly used in deep learning. The goal is to evaluate how these modifications affect performance and to understand why such improvements became standard practice.

Examples of these modifications include replacing tanh with ReLU for activation and using softmax instead of RBF for the classification layer.