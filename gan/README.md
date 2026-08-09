# Generative Adversarial Networks — Paper Reproduction

This project is a PyTorch reproduction of the original **Generative Adversarial Networks** paper by Ian Goodfellow et al. (2014).

> **Paper:** *Generative Adversarial Nets*  
> Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio  
> 2014

The goal of this reproduction is to implement the original GAN training procedure, reproduce the MNIST experiment as closely as practical, and investigate several training choices through controlled ablations.

---

## 1. Project Goals

The reproduction focuses on:

- Implementing the original GAN architecture in PyTorch.
- Reproducing the MNIST experiment.
- Using the original discriminator and generator objectives.
- Reproducing the learning-rate and momentum schedules.
- Initializing the generator output bias from the training-data pixel marginals.
- Training with uniform random latent vectors.
- Evaluating generator and discriminator losses on a validation set.
- Saving model checkpoints and generated samples.
- Recording training curves and experiment results.
- Performing ablation experiments to understand GAN training dynamics.

The original paper used Theano/Pylearn2 and GPU hardware from the time of publication. This implementation uses modern PyTorch and CPU training, so exact numerical reproduction is not expected.

---

## 2. Original GAN Objective

The discriminator is trained to maximize:

```text
E_x~pdata [log D(x)] + E_z~pz [log(1 - D(G(z)))]
```

The generator in the original minimax formulation minimizes:

```text
E_z~pz [log(1 - D(G(z)))]
```

The paper also discusses the alternative **non-saturating generator objective**:

```text
E_z~pz [-log D(G(z))]
```

This reproduction investigates the saturated/minimax generator objective as an ablation.

---

## 3. Architecture

### Generator

The MNIST generator takes a 100-dimensional latent vector and produces a 784-dimensional image vector.

```text
z ∈ R^100
    ↓
Linear(100 → 1200)
    ↓
ReLU
    ↓
Linear(1200 → 1200)
    ↓
ReLU
    ↓
Linear(1200 → 784)
    ↓
Sigmoid
    ↓
28 × 28 image
```

The generator uses a sigmoid output so generated pixel values are in approximately `[0, 1]`.

The output-layer bias is initialized using the mean pixel values of the training data, following the initialization used in the original Pylearn2 configuration.

### Discriminator

The discriminator takes a 784-dimensional MNIST image.

```text
784
 ↓
Maxout: 240 units, 5 pieces
 ↓
Maxout: 240 units, 5 pieces
 ↓
Sigmoid
 ↓
D(x) ∈ [0, 1]
```

The baseline discriminator therefore uses:

- Maxout hidden layer: 240 units, 5 pieces
- Maxout hidden layer: 240 units, 5 pieces
- Sigmoid output

---

## 4. Training Configuration

The baseline configuration follows the original MNIST configuration as closely as practical.

| Parameter | Baseline |
|---|---:|
| Dataset | MNIST |
| Latent dimension | 100 |
| Generator hidden units | 1200 → 1200 |
| Generator activation | ReLU |
| Generator output | Sigmoid |
| Discriminator hidden units | 240 → 240 |
| Discriminator activation | Maxout |
| Maxout pieces | 5 |
| Batch size | 32 |
| Initial learning rate | 0.1 |
| Initial momentum | 0.5 |
| Final momentum | 0.7 |
| Training epochs | 22 |
| Latent distribution | Uniform |
| Optimizer | SGD + Momentum |

The original Pylearn2 configuration used a batch size of 100. A batch size of 32 was used here because the reproduction was run on CPU hardware.

---

## 5. Learning-Rate Schedule

The original configuration uses exponential learning-rate decay:

```text
decay_factor = 1.000004
min_lr = 0.000001
```

The effective learning rate therefore changes very slowly during the 22-epoch experiment.

For the baseline run, the learning rate remained approximately:

```text
0.100000 → 0.099991
```

for both the generator and discriminator.

---

## 6. Momentum Schedule

The original configuration starts momentum at:

```text
0.5
```

and increases it toward:

```text
0.7
```

using a momentum adjustment period of 250 steps.

In the reproduction, the reported momentum at the end of training was approximately:

```text
0.5169
```

because only a portion of the 250-step schedule was reached during the run.

---

## 7. Validation

A separate validation set is evaluated after each epoch.

For each validation batch:

1. Generate fake samples from random latent vectors.
2. Evaluate the discriminator on real images.
3. Evaluate the discriminator on generated images.
4. Compute discriminator loss.
5. Generate another batch of fake samples.
6. Compute the generator loss.
7. Average the losses over the validation dataset.

The validation loss is used for monitoring rather than model selection.

GAN losses should not be interpreted in the same way as ordinary supervised-learning validation losses because the generator and discriminator are simultaneously changing.

---

# 8. Baseline Results

### Baseline

The baseline used the original architecture and SGD with momentum.

```text
Test Results
Discriminator: 1.8490784399091353e-06
Generator:    -2.3841860752327193e-07
```

Final epoch:

```text
Epoch 22/22:
Discriminator Train Loss: 0.0000
Generator Train Loss:    -0.0000
Discriminator Val Loss:  0.0000
Generator Val Loss:     -0.0000

D Learning Rate: 0.09999120
G Learning Rate: 0.09999120

D Momentum: 0.5169
G Momentum: 0.5169

Training time: 24m 5s
```

The discriminator became extremely confident during training. This is an important observation for this reproduction because discriminator dominance is one of the central difficulties discussed in GAN training.

---

# 9. Ablation Studies

Three ablations were performed.

## Ablation 1 — Saturated Generator Loss

The generator was trained using the original minimax/saturating objective:

```text
log(1 - D(G(z)))
```

rather than the commonly used non-saturating objective:

```text
-log(D(G(z)))
```

Results:

```text
Test Results
Discriminator: 5.566468963115767e-06
Generator:    -1.4901171425663051e-06
```

Final epoch:

```text
Discriminator Train Loss: 0.0000
Generator Train Loss:    13.3585
Discriminator Val Loss:  0.0000
Generator Val Loss:     13.4332

D Learning Rate: 0.09999120
G Learning Rate: 0.09999120

D Momentum: 0.5169
G Momentum: 0.5169

Training time: 107m 6s
```

### Observation

The saturated generator objective caused the generator loss to become very large as the discriminator became increasingly confident.

This is consistent with the well-known gradient-saturation problem of the minimax generator objective.

---

## Ablation 2 — Reduced Discriminator Capacity

The saturated generator loss was retained, while the discriminator hidden layers were reduced from:

```text
240 → 240
```

to:

```text
240 → 120
```

Results:

```text
Test Results
Discriminator: 5.794414085893554e-06
Generator:    -1.3709077393286861e-06
```

Final epoch:

```text
Discriminator Train Loss: 0.0000
Generator Train Loss:    13.6391
Discriminator Val Loss:  0.0000
Generator Val Loss:     13.5184

D Learning Rate: 0.09999120
G Learning Rate: 0.09999120

D Momentum: 0.5169
G Momentum: 0.5169

Training time: 37m 2s
```

### Observation

Reducing discriminator capacity substantially reduced training time, but it did not solve discriminator dominance.

The discriminator still became highly confident and the generator's saturated loss remained large.

This suggests that simply reducing discriminator capacity was insufficient to balance the adversarial game under this configuration.

---

## Ablation 3 — Adam Optimizer

The saturated generator objective was retained, while SGD + Momentum was replaced with Adam.

Results:

```text
Test Results
Discriminator: 1.5497212473292164e-10
Generator:    0.0
```

Final epoch:

```text
Discriminator Train Loss: 0.0000
Generator Train Loss:    26.3774
Discriminator Val Loss:  0.0000
Generator Val Loss:      26.5853

Training time: 157m 2s
```

### Observation

Adam did not improve the adversarial balance in this experiment.

In fact, the discriminator became even more dominant, while the saturated generator loss increased substantially.

This experiment demonstrates that changing the optimizer alone does not necessarily resolve GAN instability or discriminator dominance.

---

# 10. Ablation Summary

| Experiment | Main Change | Test D Loss | Test G Loss | Final G Loss | Time |
|---|---|---:|---:|---:|---:|
| Baseline | Original configuration | 1.85e-6 | -2.38e-7 | ~0 | 24m |
| Ablation 1 | Saturated generator loss | 5.57e-6 | -1.49e-6 | 13.36 | 107m |
| Ablation 2 | Saturated loss + smaller D | 5.79e-6 | -1.37e-6 | 13.64 | 37m |
| Ablation 3 | Saturated loss + Adam | 1.55e-10 | 0.0 | 26.38 | 157m |

---

# 11. Key Findings

### 1. The discriminator dominates the training process

Across the experiments, discriminator losses rapidly approach zero.

This means the discriminator becomes extremely confident in distinguishing real training examples from generated samples.

This is especially clear in the saturated-loss experiments.

### 2. The saturated generator objective exposes gradient saturation

When the discriminator becomes too successful:

```text
D(G(z)) → 0
```

and the generator's minimax objective:

```text
log(1 - D(G(z)))
```

approaches zero in value but provides very weak gradients.

The reported generator loss can nevertheless become numerically large depending on the exact implementation/sign convention and evaluation being used.

The important observation is that the generator is not receiving a sufficiently useful learning signal once the discriminator becomes too confident.

### 3. Reducing discriminator capacity helped runtime but not balance

The 240 → 120 discriminator reduced training time considerably:

```text
107m → 37m
```

for the saturated-loss experiment.

However, discriminator dominance remained.

### 4. Adam did not automatically fix the problem

The Adam experiment took considerably longer and produced an even stronger discriminator.

This shows that optimizer choice alone is not enough to guarantee stable GAN training.

### 5. GAN losses are not directly comparable to ordinary supervised losses

A low discriminator loss does not mean the GAN is necessarily producing good samples.

The generator and discriminator are competing objectives, so generated samples and training dynamics should also be inspected.

---

# 12. Important Paper Connection

The original paper explicitly discusses the requirement that the generator and discriminator remain synchronized during training.

In particular, the paper warns that the generator should not be trained too much without updating the discriminator, because this can lead to **mode collapse**, described in the paper as the "Helvetica scenario."

This is an important interpretation of the experiments in this reproduction:

```text
G and D must remain balanced.
```

If the discriminator becomes too strong, the generator can receive a poor gradient signal.

Conversely, if the generator is updated too aggressively relative to the discriminator, the discriminator may become stale.

---

# 13. Generated Samples

Generated samples are periodically saved during training to inspect qualitative progress.

Sample inspection is important because the GAN losses alone do not fully describe generator quality.

For each experiment, compare:

- Early generated samples
- Middle-training samples
- Final generated samples
- Diversity between generated samples
- Sharpness of generated digits
- Evidence of mode collapse

---

# 14. Checkpointing

Training checkpoints should contain at minimum:

```text
generator_state_dict
discriminator_state_dict
generator_optimizer_state_dict
discriminator_optimizer_state_dict
epoch
training history
```

This allows experiments to be resumed and makes it possible to preserve the exact model state associated with generated samples.

---

# 15. Project Structure

A possible project structure is:

```text
gan/
├── models/
│   ├── generator.py
│   ├── discriminator.py
│   ├── losses.py
│   ├── train.py
│   ├── evaluate.py
│   ├── preprocessing.py
│   └── config.py
│
├── experiments/
│   ├── trial_0.py
│   ├── ablation_1.py
│   ├── ablation_2.py
│   └── ablation_3.py
│
├── notebooks/
│   └── results.ipynb
│
├── checkpoints/
├── samples/
└── README.md
```

---

# 16. Reproducibility Notes

Exact reproduction of the 2014 paper is not expected.

The original implementation used:

- Theano
- Pylearn2
- NVIDIA GeForce GTX-580 GPUs
- The software versions available around 2014

This reproduction instead uses:

- PyTorch
- Modern Python
- CPU training
- Modern numerical libraries

Differences in:

- Random number generation
- Weight initialization
- Floating-point operations
- CPU/GPU computation
- Batch size
- Framework implementation
- Optimizer implementation

can all affect the final result.

Therefore, this project should be viewed as a **modern implementation and reproduction study**, rather than an exact bit-for-bit reproduction.

---

# 17. Reference Implementation

The original authors' repository used Pylearn2/Theano configuration files such as:

```text
mnist.yaml
pretrain.yaml
```

and included a Parzen-window log-likelihood evaluation script.

The original code is useful as a reference for understanding:

- Network architecture
- Initialization
- Training hyperparameters
- Momentum scheduling
- Learning-rate decay
- Generator pretraining
- Evaluation methodology

---

# 18. Conclusion

This reproduction demonstrates several important properties of the original GAN framework.

The baseline successfully trains a generator and discriminator, but the discriminator quickly becomes extremely confident.

The ablation studies further demonstrate that:

- The saturated minimax generator objective can produce difficult training dynamics.
- Reducing discriminator capacity improves runtime but does not necessarily balance training.
- Switching to Adam does not automatically solve discriminator dominance.
- GAN training requires careful coordination between generator and discriminator updates.

These results reinforce one of the central lessons of the original GAN paper:

> Successful GAN training depends on maintaining a useful adversarial game between the generator and discriminator rather than optimizing either network independently.

The project therefore serves both as a paper reproduction and as an investigation into the practical training dynamics of the original GAN formulation.
