# Project Specification

In this project, I will reproduce the results of the original GAN model presented in *Generative Adversarial Nets* (Goodfellow et al., 2014). The objective is to faithfully reproduce the authors' MNIST experiment by implementing the model in PyTorch while following the architecture and hyperparameters provided in the official implementation.

---

# Baseline Model

The baseline model follows the official MNIST configuration released by the authors.

## Dataset

* **Dataset:** MNIST
* **Training Set:** 50,000 images (indices 0–49,999)
* **Validation Set:** 10,000 images (indices 50,000–59,999)
* **Test Set:** Standard MNIST test set (10,000 images)
* **Image Size:** 28 × 28 pixels
* **Channels:** 1 (grayscale)

### Data Preprocessing

* Convert images to PyTorch tensors using `transforms.ToTensor()`.
* Pixel values are automatically scaled from **[0, 255]** to **[0, 1]**.
* No additional normalization is performed, as the generator uses a **Sigmoid** output layer.

---

## Latent Space

The generator receives a randomly sampled latent vector as input.

| Parameter          |   Value |
| ------------------ | ------: |
| Latent Dimension   |     100 |
| Noise Distribution | Uniform |

---

## Generator Architecture

The generator is implemented as a multilayer perceptron (MLP).

| Layer  | Output Size | Activation    |
| ------ | ----------: | ------------- |
| Input  |         100 | Uniform Noise |
| Linear |        1200 | ReLU          |
| Linear |        1200 | ReLU          |
| Linear |         784 | Sigmoid       |
| Output | 1 × 28 × 28 | Reshape       |

### Notes

* Input consists of a 100-dimensional latent vector sampled from a uniform distribution.
* Two hidden fully connected layers each contain 1200 neurons.
* The output layer contains 784 neurons corresponding to a flattened MNIST image.
* The output is reshaped into a 28 × 28 grayscale image.
* The output layer bias is initialized using the marginal pixel statistics of the MNIST training set, matching the official implementation.

---

## Discriminator Architecture

The discriminator is implemented as a multilayer perceptron using Maxout hidden layers.

| Layer  | Configuration         |
| ------ | --------------------- |
| Input  | 784 (flattened image) |
| Maxout | 240 units, 5 pieces   |
| Maxout | 240 units, 5 pieces   |
| Linear | 1                     |
| Output | Sigmoid               |

### Notes

* Input images are flattened before entering the network.
* Each Maxout layer contains 240 output units, where each unit computes the maximum over 5 learned affine feature maps.
* The final Sigmoid layer outputs the probability that an input image is real.

---

## Training Configuration

| Hyperparameter         |                             Value |
| ---------------------- | --------------------------------: |
| Optimizer              | Stochastic Gradient Descent (SGD) |
| Learning Rate          |                               0.1 |
| Batch Size             |                                32 |
| Initial Momentum       |                               0.5 |
| Final Momentum         |                               0.7 |
| Learning Rate Schedule |                 Exponential Decay |
| Latent Dimension       |                               100 |
| Noise Distribution     |                           Uniform |

### Additional Training Details

* Training alternates between updating the discriminator and the generator.
* The discriminator uses the dropout configuration defined in the original implementation through `AdversaryCost2`.
* Model checkpoints are saved after each epoch.

---


## Evaluation

The baseline and ablation models will be evaluated using the Parzen-window
log-likelihood estimation procedure used in the original repository.

### Sample Generation

- Generate 10,000 samples from the trained generator.
- Use the generated samples to construct a Parzen-window density estimator.

### Validation

- Use the 10,000-image MNIST validation split (images 50,000–59,999)
  to select the Parzen-window bandwidth σ.
- Evaluate a range of candidate σ values from 0.1 to 1.0 using logarithmic spacing.
- Select the σ producing the highest validation log-likelihood.

### Test Evaluation

- Evaluate the selected Parzen estimator on the official 10,000-image
  MNIST test set.
- Report the mean test log-likelihood.
- Report the standard error of the estimated log-likelihood.

### Qualitative Evaluation

In addition to the quantitative evaluation, generated MNIST samples will
be visualized at different stages of training to assess sample quality and
training progression.


---

# Reproduction Notes

This project aims to reproduce the original 2014 GAN implementation rather than a modern GAN baseline. Consequently, several design choices differ from contemporary implementations.

* The generator outputs images using a **Sigmoid** activation rather than **Tanh**.
* Training uses **SGD with momentum** instead of Adam.
* The discriminator uses **Maxout** hidden layers instead of LeakyReLU.
* Latent vectors are sampled from a **uniform distribution** instead of a Gaussian distribution.
* The architecture follows the official implementation released by the authors.
