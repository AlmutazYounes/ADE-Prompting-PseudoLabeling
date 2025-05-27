# ADE Extraction Approaches Comparison

Generated on: 2025-05-25 02:51:02

## Summary of Results

### Overall F1 Scores

| Approach | Overall F1 | Drug F1 | ADE F1 |
|----------|------------|---------|--------|
| Direct LLM | 0.798 | 0.973 | 0.629 |
| DSPy LLM | 0.802 | 0.953 | 0.633 |
| ModernBERT (Direct) | 0.075 | 0.071 | 0.013 |
| ModernBERT (DSPy) | 0.064 | 0.051 | 0.041 |

### Detailed Metrics

#### Direct LLM

| Entity Type | Precision | Recall | F1 |
|-------------|-----------|--------|----|
| Overall | 0.771 | 0.839 | 0.798 |
| Drug | 0.960 | 1.000 | 0.973 |
| Ade | 0.605 | 0.675 | 0.629 |

#### DSPy LLM

| Entity Type | Precision | Recall | F1 |
|-------------|-----------|--------|----|
| Overall | 0.808 | 0.816 | 0.802 |
| Drug | 0.940 | 0.980 | 0.953 |
| Ade | 0.625 | 0.655 | 0.633 |

#### ModernBERT (Direct)

| Entity Type | Precision | Recall | F1 |
|-------------|-----------|--------|----|
| Overall | 0.047 | 0.229 | 0.075 |
| Drug | 0.043 | 0.220 | 0.071 |
| Ade | 0.008 | 0.040 | 0.013 |

#### ModernBERT (DSPy)

| Entity Type | Precision | Recall | F1 |
|-------------|-----------|--------|----|
| Overall | 0.043 | 0.150 | 0.064 |
| Drug | 0.040 | 0.130 | 0.051 |
| Ade | 0.026 | 0.120 | 0.041 |

