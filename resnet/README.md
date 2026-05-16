# ResNet Paper Reproduction (PyTorch)

Reproduction of the paper:

> He et al., *Deep Residual Learning for Image Recognition* (2015)

This project implements and trains a ResNet-18 model from scratch using PyTorch and evaluates architectural design choices through controlled experiments on CIFAR-10.

---

# Project Goals

The purpose of this reproduction was to:

* Understand residual learning and skip connections
* Implement ResNet-18 from scratch using `torch.nn`
* Build a full training/evaluation pipeline in PyTorch
* Validate implementation correctness
* Run controlled architecture experiments
* Analyze the effect of ResNet design choices on CIFAR-10

---

# Paper

**Deep Residual Learning for Image Recognition**

Authors:

* Kaiming He
* Xiangyu Zhang
* Shaoqing Ren
* Jian Sun

Year:

* 2015

Main contribution:

* Introduction of residual learning using skip connections to enable optimization of deeper neural networks.

---

# Architecture Implemented

## BasicBlock

Implemented components:

* 3x3 convolutions
* Batch Normalization
* ReLU activations
* Identity skip connections
* Projection shortcuts for dimension changes

## ResNet-18

Implemented:

* Stem layer
* 4 residual stages
* Global average pooling
* Fully connected classifier

Configuration:

```text
[2, 2, 2, 2]
```

---

# Dataset

Dataset used:

* CIFAR-10

Preprocessing:

```python
RandomCrop(32, padding=4)
RandomHorizontalFlip()
Normalize(mean, std)
```

---

# Training Setup

Optimizer:

```python
SGD(
    lr=0.1,
    momentum=0.9,
    weight_decay=1e-4
)
```

Scheduler:

```python
StepLR(step_size=5, gamma=0.1)
```

Loss:

```python
CrossEntropyLoss
```

Device:

```text
CPU
```

---

# Validation Checks

Several correctness checks were performed.

## Shape Validation

Verified tensor shapes through:

* residual blocks
* downsampling layers
* full forward pass

## Overfit Small Batch Test

A fixed batch was trained repeatedly until the model memorized it.

Result:

```text
100% training accuracy achieved
```

This confirmed:

* gradients flow correctly
* optimizer updates work
* model implementation is trainable

---

# Experiments

## Experiment 1 — Baseline ImageNet Stem

Configuration:

```text
7x7 convolution
stride = 2
maxpool enabled
```

Results:

```text
Test Accuracy: 69.76%
Runtime: 53m 12s
```

Observation:

The ImageNet-style stem aggressively downsamples 32x32 CIFAR-10 images, limiting performance.

---

## Experiment 2 — CIFAR-10 Stem

Configuration:

```text
3x3 convolution
stride = 1
no maxpool
```

Results:

```text
Test Accuracy: 83.29%
Runtime: 239m 15s
```

Observation:

Preserving spatial resolution significantly improved performance on CIFAR-10.

Tradeoff:

* much higher computational cost
* slower training on CPU

Conclusion:

The ImageNet stem is poorly suited for small 32x32 inputs.

---

## Experiment 3 — No Skip Connections Ablation

Configuration:

```text
CIFAR stem
skip connections disabled
```

Results:

```text
Test Accuracy: 81.35%
Runtime: 209m 7s
```

Observation:

Removing skip connections reduced performance.

Conclusion:

Residual connections improved optimization and generalization, though the effect was moderate at ResNet-18 depth.

---

# Final Results Summary

| Experiment                       | Test Accuracy | Runtime  |
| -------------------------------- | ------------- | -------- |
| ImageNet Stem                    | 69.76%        | 53m 12s  |
| CIFAR Stem                       | 83.29%        | 239m 15s |
| CIFAR Stem + No Skip Connections | 81.35%        | 209m 7s  |

---

# Key Takeaways

## 1. Residual Learning Works

Skip connections improved optimization and final performance.

## 2. Architecture Must Match Input Scale

The CIFAR-specific stem dramatically improved accuracy compared to the original ImageNet-style stem.

## 3. Validation Matters

The overfit-small-batch test was critical for debugging and verifying correctness.

## 4. Runtime Tradeoffs Matter

Preserving higher spatial resolution improved accuracy but substantially increased CPU training time.

---

# Skills Practiced

* PyTorch `torch.nn`
* CNN architecture implementation
* Residual networks
* Training loops
* Evaluation pipelines
* Experiment design
* Ablation studies
* Debugging deep learning models
* Training dynamics analysis

---

# Future Improvements

Potential extensions:

* ResNet-34 / ResNet-50
* Bottleneck blocks
* Cosine annealing scheduler
* Mixed precision training
* GPU acceleration
* Feature map visualization
* Gradient visualization
* CIFAR-100 experiments

---

# References

He, K., Zhang, X., Ren, S., & Sun, J. (2015).
*Deep Residual Learning for Image Recognition*.
