| Router | Macro-F1 | Accuracy | Avg latency (ms) | Escalation |
|---|---|---|---|---|
| Stage 1 only (TF-IDF+LogReg) | 0.923 | 0.925 | 0.58 | 0% |
| Stage 2 only (emb+MLP) | 0.957 | 0.956 | 13.19 | 100% |
| **Cascade @τ=0.54** | **0.969** | 0.969 | **1.29** | 6% |
| Cascade @τ=0.78 (max-F1) | 0.970 | 0.969 | 1.92 | 11% |

Stage-1 ECE raw=0.385 -> calibrated=0.040 (T=0.26). Cascade matches Stage-2 F1 (0.957) at 1.29 ms, a 90% average-latency reduction vs always running the encoder.
