| Experiment            | Train Loss | Validation Loss | Test Loss | Training Time | Notes |
|-----------------------|-----------:|----------------:|----------:|--------------:|-------|
| Baseline              |     0.1834 |          0.3127 |    0.3052 |       57m 57s | Reference implementation with stable convergence across all epochs. |
| Paper Optimizer       |     0.1252 |          0.2926 |    0.2818 |       70m 20s | Improved optimization and generalization over the baseline. |
| Increased d_model     |     0.0905 |          0.2455 | **0.2339** |      216m 43s | Best overall performance, but required substantially more training time. |
| Increased NUM_LAYERS  |     0.1293 |          0.2911 |    0.2784 |      139m 08s | Marginal improvement over the optimizer experiment despite nearly doubling the training time. |
| Applied Dropout       |     0.1280 |          0.2739 |    0.2643 |       97m 58s | Improved generalization with only a moderate increase in training time. |
| All Changes Applied   |     0.0950 |          0.2513 |    0.2396 |      274m 30s | Second-best performance, but the highest computational cost and no improvement over the increased d_model experiment. |