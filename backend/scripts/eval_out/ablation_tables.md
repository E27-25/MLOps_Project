### Study A — classifier head on frozen all-MiniLM-L6-v2 (5-fold CV)

| Head | Macro-F1 | Accuracy |
|---|---|---|
| MLP (128,64) early-stop [deployed] | 0.707±0.265 | 0.744 |
| MLP (128,64) no early-stop | 0.950±0.037 | 0.950 |
| MLP (256,) no early-stop | 0.950±0.038 | 0.950 |
| MLP (64,) no early-stop | 0.956±0.033 | 0.956 |
| LogReg | 0.943±0.048 | 0.944 |
| Linear SVM | 0.942±0.048 | 0.944 |
| kNN (k=5, cosine) | 0.943±0.033 | 0.944 |
| NearestCentroid (cosine) | 0.936±0.044 | 0.938 |

### Study B — embedding backbone (head = LogReg, 5-fold CV)

| Backbone | Dim | Macro-F1 | Accuracy |
|---|---|---|---|
| all-MiniLM-L6-v2 | 384 | 0.943±0.048 | 0.944 |
| all-MiniLM-L12-v2 | 384 | 0.942±0.048 | 0.944 |
| paraphrase-multilingual-MiniLM-L12-v2 | 384 | 0.942±0.055 | 0.944 |
| bge-small-en-v1.5 | 384 | 0.969±0.040 | 0.969 |
