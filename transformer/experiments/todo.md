| Trial | Change                                     | Keep Everything Else Fixed | Goal                            |
| ----: | ------------------------------------------ | -------------------------- | ------------------------------- |
|   ✅ 0 | Baseline                                   | —                          | Reference model                 |
|   ✅ 1 | Original paper LR schedule + Adam settings | Yes                        | Test optimizer                  |
| ✅ 2 | Increase `d_model` (128 → 256)             | Yes                        | Test model capacity             |
| ✅ 3 | Increase encoder/decoder layers (2 → 4)    | Yes                        | Test model depth                |
| **4** | Increase d_ff (512 → 1024)           | Yes                        | Test attention granularity      |
| **5** | Add dropout (e.g., 0.1)                    | Yes                        | Test regularization             |
| **6** | Combine the best settings                  | Yes                        | Evaluate cumulative improvement |
