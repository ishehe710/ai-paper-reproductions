| Experiment | Train Loss | Validation Loss | Test Loss | Training Time  |  Notes |
|------------|-----------:|----------------:|----------:| --------------:| -------|
| Baseline | 0.1834 | 0.3127 | 0.3052 | 57m 57s | Good translations on short/simple sentences. |
| Paper Optimizer | 0.1252 | 0.2926 | 0.2818 | 70m 20s | Lower loss with improved optimization and LR schedule. |
| Increased d_model | 0.0905 | 0.2455 | 0.2339 | 216m 43s | Best overall performance, but ~3× longer training time. |
| Increased NUM_LAYERS | 0.1293 | 0.2911 | 0.2784 | 141m 24s | Only slightly better than baseline, but takes significantly less time to train