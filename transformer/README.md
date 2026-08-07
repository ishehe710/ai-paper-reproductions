# Transformer: Attention Is All You Need (2017)

A PyTorch reproduction of the Transformer architecture introduced in **Attention Is All You Need** by Vaswani et al. (2017). This project was implemented from scratch as part of my AI Paper Reproductions series, with the goal of understanding the architecture, training procedure, and design decisions behind modern sequence-to-sequence models.

---

## Overview

The objective of this project was to reproduce the original Transformer architecture while gaining a deep understanding of its core components:

- Token embeddings
- Sinusoidal positional encoding
- Multi-head self-attention
- Encoder-decoder attention
- Position-wise feed-forward networks
- Residual connections
- Layer normalization
- Autoregressive decoding
- Training with teacher forcing
- Learning rate scheduling described in the paper

Rather than relying on PyTorch's built-in `nn.Transformer`, every major component was implemented manually.

---

## Repository Structure

```
transformer/
│
├── data/
│   ├── dataset.py
│   ├── dataloader.py
│   └── tokenizer.py
│
├── model/
│   ├── attention.py
│   ├── decoder.py
│   ├── decoder_layer.py
│   ├── embedding.py
│   ├── encoder.py
│   ├── encoder_layer.py
│   ├── feed_forward.py
│   ├── multi_head.py
│   ├── positional.py
│   ├── train.py
│   ├── evaluate.py
│   └── transformer.py
│
├── experiments/
│   ├── baseline.ipynb
│   ├── optimizer.ipynb
│   ├── d_model.ipynb
│   ├── num_layers.ipynb
│   ├── dropout.ipynb
│   └── combined.ipynb
│
└── README.md
```

---

## Dataset

The model was trained on an English–French parallel corpus consisting of approximately **175,000 sentence pairs**.

Due to hardware limitations, experiments were conducted on a reproducible subset of:

- Training: **4,096 samples**
- Validation: **512 samples**
- Testing: **512 samples**

A fixed random seed was used to ensure consistent train/validation/test splits across all experiments.

---

## Baseline Model

The baseline implementation consisted of:

- 2 Encoder Layers
- 2 Decoder Layers
- 8 Attention Heads
- d_model = 128
- d_ff = 512
- Adam optimizer
- Fixed learning rate
- No dropout

This baseline served as the reference implementation for the ablation study.

---

## Experiments

The following experiments were conducted to evaluate individual architectural and optimization choices.

| Experiment | Train Loss | Validation Loss | Test Loss | Training Time | Notes |
|------------|-----------:|----------------:|----------:|--------------:|-------|
| Baseline | 0.1834 | 0.3127 | 0.3052 | 57m 57s | Reference implementation with stable convergence. |
| Paper Optimizer | 0.1252 | 0.2926 | 0.2818 | 70m 20s | Transformer learning-rate schedule and Adam parameters from the paper. |
| Increased d_model | 0.0905 | 0.2455 | **0.2339** | 216m 43s | Best overall performance. |
| Increased Layers | 0.1293 | 0.2911 | 0.2784 | 139m 08s | Marginal improvement despite increased computational cost. |
| Dropout | 0.1280 | 0.2739 | 0.2643 | 97m 58s | Improved generalization with moderate overhead. |
| Combined Model | 0.0950 | 0.2513 | 0.2396 | 274m 30s | Second-best performance but longest training time. |

---

## Key Findings

The ablation study produced several notable observations:

- The Transformer learning-rate schedule significantly improved convergence over a fixed learning rate.
- Increasing the model dimension (`d_model`) produced the largest improvement in translation quality.
- Increasing the number of encoder and decoder layers resulted in only modest gains while substantially increasing training time.
- Applying dropout improved validation and test performance by reducing overfitting.
- Combining all architectural modifications did not outperform increasing `d_model` alone, suggesting diminishing returns under the reduced dataset size.

Overall, increasing model width proved more beneficial than increasing model depth for this reproduction.

---

## Example Translations

| English | Ground Truth | Model Output |
|----------|--------------|--------------|
| Are we finished? | En avons-nous terminé ? | En demain terminé ? |
| I just want to say I'm sorry. | Je veux juste dire que je suis désolé. | Je veux juste dire que je suis désolé. |
| You're a man now. | Vous êtes désormais un homme. | Vous êtes désormais un homme. |
| It's a great night to go dancing. | C'est une belle nuit pour aller danser. | C'est une belle nuit pour aller danser. |

Although the model occasionally substituted semantically related words, it generally produced fluent translations on shorter and simpler sentences.

---

## Limitations

To keep experimentation feasible on CPU hardware, the reproduced architecture differs from the original Transformer presented in the paper.

Compared to the original model:

| Parameter | Original Paper | This Project |
|-----------|---------------:|-------------:|
| Encoder Layers | 6 | 2 (4 in experiments) |
| Decoder Layers | 6 | 2 (4 in experiments) |
| d_model | 512 | 128 (256 in experiments) |
| d_ff | 2048 | 512 |
| Training Data | Millions of sentence pairs | 4,096 training samples |

Consequently, the reported results should be viewed as a scaled reproduction rather than an exact replication of the original model.

---

## Future Work

Future improvements include:

- Training on the full dataset
- Reproducing the complete Transformer Base architecture
- GPU training
- Beam search decoding
- Label smoothing
- BLEU score evaluation
- Attention visualization
- Mixed precision training
- Larger-scale hyperparameter exploration

---

## Paper

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin.

**Attention Is All You Need**

NeurIPS 2017.

---

## Technologies

- Python
- PyTorch
- NumPy
- Jupyter Notebook

---

## Author

**Adam Shehe**

This project is part of my **AI Paper Reproductions** portfolio, where I implement influential machine learning papers from scratch to better understand their underlying algorithms and engineering design.