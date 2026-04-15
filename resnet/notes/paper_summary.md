# Deep Residual Learning for Image Recognition (He et al., 2015)
## Implementation-Focused Summary

---

## 1. Core Idea

Instead of directly learning a mapping:

    H(x)

ResNet reformulates the problem as learning a residual:

    F(x) = H(x) - x

So the original function becomes:

    H(x) = F(x) + x

### Key Insight
- It is easier to learn a residual (difference from identity) than a full transformation.
- If identity is optimal → network can learn F(x) = 0.
- This makes optimization of deep networks significantly easier.

---

## 2. The Degradation Problem

Observation from the paper:
- As depth increases:
  - Training error should decrease (more capacity)
  - But instead:
    - Accuracy saturates
    - Then degrades

### Important:
- This is NOT overfitting
- This is an **optimization problem**

---

## 3. Why Residual Connections Work

### Without Residuals:
- Deep networks suffer from:
  - vanishing gradients
  - exploding gradients
  - optimization difficulty

### With Residuals:
- Skip connection allows:
  - direct gradient flow
  - easier identity mapping
  - stable training for very deep networks

Gradient can bypass layers through identity path.

---

## 4. Residual Block Structure (BasicBlock)

Standard block:

    x
    │
    ├── Conv → BN → ReLU → Conv → BN ──┐
    │                                  │
    └──────────── identity ─────────────┘
                       ↓
                     Add
                       ↓
                     ReLU

### Important Details:
- Addition happens BEFORE final ReLU
- No ReLU after second BN before addition
- Final activation applied AFTER addition

---

## 5. Skip Connections (Shortcut Types)

### 1. Identity Shortcut
Used when:
- Input and output dimensions match

    y = F(x) + x

---

### 2. Projection Shortcut (1×1 Conv)
Used when:
- Dimensions differ (channels or spatial size)

    y = F(x) + W_s x

Where:
- W_s is a 1×1 convolution

Used for:
- Downsampling (stride > 1)
- Increasing channels

---

## 6. Residual Function F(x)

In practice:
- F(x) is a stack of layers:

BasicBlock:
    Conv(3x3) → BN → ReLU → Conv(3x3) → BN

Bottleneck (deeper networks):
    1x1 → 3x3 → 1x1

---

## 7. Network Architecture (ResNet-18 / 34 style)

General structure:

    Input
    → Conv(7x7) → BN → ReLU → MaxPool
    → Layer1
    → Layer2 (downsample)
    → Layer3 (downsample)
    → Layer4 (downsample)
    → Global Average Pool
    → Fully Connected

### Layer Pattern:
- Stack multiple residual blocks
- Each stage increases channels:
    64 → 128 → 256 → 512

---

## 8. Downsampling Strategy

Occurs when:
- Moving between stages

Implemented via:
- stride=2 in first conv of block
- AND projection shortcut (1×1 conv)

---

## 9. FLOPs (Clarification)

- FLOPs = number of floating point operations
- Measures computational cost
- NOT number of parameters

---

## 10. Key Implementation Details (IMPORTANT)

When coding:

### Block:
- Two conv layers
- BatchNorm after each conv
- ReLU after first conv AND after addition

### Forward Pass:
1. Save identity (x)
2. Compute F(x)
3. If needed → transform identity using 1×1 conv
4. Add:
       out += identity
5. Apply ReLU

---

## 11. What to Be Careful About

- Shape mismatches (most common bug)
- Forgetting projection shortcut
- Wrong placement of ReLU
- Incorrect stride handling

---

## 12. What You Should Understand After This

- Why deeper networks fail without residuals
- Why identity mapping is critical
- How skip connections help gradient flow
- How modern CNN architectures are built

---

## 13. Minimal Mental Model

ResNet block =

    output = ReLU( F(x) + x )

That’s it.

Everything else is engineering details.