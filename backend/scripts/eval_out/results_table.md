| Method | Macro-F1 | Accuracy | Latency (ms) | Model size |
|---|---|---|---|---|
| Majority | 0.053 | 0.188 | — | — |
| Keyword/regex | 0.817 | 0.825 | 0.0 | — |
| TF-IDF + LogReg | 0.948±0.039 | 0.950 | 0.8 | 147 KB |
| Centroid (cosine) | 0.936±0.044 | 0.938 | 13.8 | 10 KB |
| kNN (k=5) | 0.943±0.033 | 0.944 | 15.0 | 242 KB |
| LogReg (emb) | 0.943±0.048 | 0.944 | 13.8 | 19 KB |
| MLP (128,64) early-stop | 0.707±0.265 | 0.744 | 13.9 | 687 KB |
| MLP (64,) [ours] | 0.956±0.033 | 0.956 | 14.0 | 405 KB |
