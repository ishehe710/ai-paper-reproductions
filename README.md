# AI Paper Reproductions

A structured collection of deep learning and machine learning paper reproductions implemented primarily using PyTorch.

This repository focuses specifically on:

* reproducing influential AI/ML papers from scratch
* understanding neural network architectures
* analyzing training behavior and optimization
* validating implementations through experiments
* developing strong PyTorch and deep learning engineering skills

The goal is not simply to replicate published results, but to understand:

* why architectures work
* how training dynamics behave
* how design choices affect optimization and performance
* how modern deep learning systems are implemented internally

---

# Repository Goals

This repository is dedicated entirely to paper reproductions.

Each project aims to:

* read and analyze the original paper
* implement the architecture manually
* reproduce core ideas experimentally
* validate correctness through debugging and testing
* run controlled experiments and ablations
* compare historical and modern design choices

The reproductions prioritize:

```text
understanding > benchmark chasing
```

---

This repository focuses on:

## 1. Paper Reproduction

Reimplement influential machine learning and deep learning papers from scratch, within torch.nn.

## 2. Deep Learning Understanding

Develop intuition for:

* optimization
* architecture design
* training stability

## 3. Experimentation & Validation

Each project emphasizes:

* controlled experiments
* ablation studies
* debugging
* correctness checks
* training analysis

---

# Repository Structure

```text
ai-paper-reproductions/
│
├── lenet-5/
├── resnet/
└── README.md
```

Each project generally contains:

* architecture implementations
* notebooks and experiments
* paper notes
* debugging notes
* evaluation results
* project-specific documentation

---

# Tech Stack

Core tools used throughout the repository:

* Python
* PyTorch
* torchvision
* NumPy
* Matplotlib
* Jupyter Notebooks


---

# Project Roadmap

## Completed Projects

## LeNet-5 Reproduction (1998)

Paper:

> LeCun et al., *Gradient-Based Learning Applied to Document Recognition*

Implemented:

* original LeNet-5 architecture
* scaled tanh activations
* RBF classifier
* multiple modern variants

Experiments included:

* ReLU vs tanh
* AvgPool vs MaxPool
* SGD vs Adam
* RBF vs Softmax classifier
* BatchNorm variants

Key Results:

| Experiment       | Test Accuracy |
| ---------------- | ------------- |
| Baseline LeNet-5 | 95.78%        |
| Modern LeNet     | 99.27%        |

Main takeaway:

Modern deep learning components significantly improve the original architecture when adapted carefully.

Project README: fileciteturn5file0

---

## ResNet Reproduction (2015)

Paper:

> He et al., *Deep Residual Learning for Image Recognition*

Implemented:

* BasicBlock
* ResNet-18
* training/evaluation pipeline
* CIFAR-10 experiments
* residual connection ablations
* CIFAR-specific stem experiments

Key Results:

| Experiment                       | Test Accuracy |
| -------------------------------- | ------------- |
| ImageNet Stem                    | 69.76%        |
| CIFAR Stem                       | 83.29%        |
| CIFAR Stem + No Skip Connections | 81.35%        |

Main takeaways:

* preserving spatial resolution significantly improves CIFAR-10 performance
* skip connections improve optimization and generalization
* architecture design must match input scale

Project README: fileciteturn5file1

---

# Projects In Progress

## Transformer Reproduction (2017)

Paper:

> Vaswani et al., *Attention Is All You Need*

Current focus:

* self-attention
* multi-head attention
* positional encoding
* encoder-decoder architecture
* transformer training dynamics

Primary learning goal:

```text
attention mechanisms and LLM foundations
```

---

# Planned Projects

## Tier 3 — Additional Paper Reproductions

### GAN Reproduction (2014)

Focus areas:

* adversarial training
* generator/discriminator dynamics
* generative modeling

---

# Future Paper Reproduction Possibilities

Potential future reproductions in priority order:

1. Diffusion Models (2020)
2. DQN (Deep Q-Networks)

---

# Learning Philosophy

This repository prioritizes:

```text
understanding > benchmark chasing
```

The emphasis is on:

* implementation clarity
* experimentation
* debugging
* architectural understanding
* engineering discipline

rather than solely maximizing leaderboard performance.

---

# Long-Term Goal

The broader goal of this repository is to transition from:

```text
deep learning research reproduction
→ ML engineering
→ scalable ML systems
→ MLOps
```

while building strong practical understanding of modern AI systems.
