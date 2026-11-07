## Retrieval Evaluation Report

### Overview

Retrieval performance was evaluated across four datasets containing **200 questions**. Baseline retrieval was compared against retrieval with a Cross-Encoder reranker.

| Dataset | Questions | Baseline MRR | Reranked MRR | Improvement |
|---|---:|---:|---:|---:|
| Single-paper | 80 | 68.85% | 79.57% | +10.73 pp |
| Multi-paper | 50 | 71.30% | 87.14% | +15.83 pp |
| Numerical evidence-specific | 40 | 42.07% | 44.46% | +2.39 pp |
| Architecture & methodology | 30 | 39.81% | 48.26% | +8.44 pp |
| **Total** | **200** | — | — | — |

*Metrics are macro-averaged across questions. Improvement is expressed in percentage points (pp).*

### 1. Single-Paper Retrieval

**Dataset size:** 80 questions

| Metric | Baseline | Reranked | Δ (pp) |
|---|---:|---:|---:|
| MRR | 68.85% | 79.57% | +10.73 |
| Precision@1 | 53.75% | 67.50% | +13.75 |
| Recall@1 | 18.13% | 23.13% | +5.00 |
| nDCG@1 | 46.96% | 61.79% | +14.82 |
| Precision@3 | 38.33% | 48.33% | +10.00 |
| Recall@3 | 39.17% | 49.38% | +10.21 |
| nDCG@3 | 44.66% | 58.50% | +13.84 |
| Precision@5 | 30.75% | 34.25% | +3.50 |
| Recall@5 | 52.29% | 58.96% | +6.67 |
| nDCG@5 | 50.91% | 62.27% | +11.36 |
| Precision@10 | 20.50% | 21.88% | +1.38 |
| Recall@10 | 70.42% | 75.00% | +4.58 |
| nDCG@10 | 57.92% | 68.02% | +10.09 |

### 2. Multi-Paper Retrieval

**Dataset size:** 50 questions

| Metric | Baseline | Reranked | Δ (pp) |
|---|---:|---:|---:|
| MRR | 71.30% | 87.14% | +15.83 |
| Precision@1 | 58.00% | 84.00% | +26.00 |
| Recall@1 | 29.00% | 42.00% | +13.00 |
| nDCG@1 | 58.00% | 84.00% | +26.00 |
| Precision@2 | 47.00% | 42.00% | −5.00 |
| Recall@2 | 47.00% | 42.00% | −5.00 |
| nDCG@2 | 49.49% | 51.50% | +2.02 |
| Precision@3 | 34.00% | 32.00% | −2.00 |
| Recall@3 | 51.00% | 48.00% | −3.00 |
| nDCG@3 | 51.94% | 55.18% | +3.24 |
| Precision@5 | 28.80% | 20.00% | −8.80 |
| Recall@5 | 72.00% | 50.00% | −22.00 |
| nDCG@5 | 62.66% | 56.13% | −6.52 |
| Precision@10 | 17.20% | 13.60% | −3.60 |
| Recall@10 | 86.00% | 68.00% | −18.00 |
| nDCG@10 | 68.26% | 63.08% | −5.18 |

### 3. Numerical Evidence-Specific Retrieval

**Dataset size:** 40 questions

| Metric | Baseline | Reranked | Δ (pp) |
|---|---:|---:|---:|
| MRR | 42.07% | 44.46% | +2.39 |
| Precision@1 | 32.50% | 32.50% | 0.00 |
| Recall@1 | 32.50% | 32.50% | 0.00 |
| nDCG@1 | 32.50% | 32.50% | 0.00 |
| Precision@3 | 15.83% | 19.17% | +3.33 |
| Recall@3 | 47.50% | 57.50% | +10.00 |
| nDCG@3 | 41.96% | 46.96% | +5.00 |
| Precision@5 | 10.50% | 12.50% | +2.00 |
| Recall@5 | 52.50% | 62.50% | +10.00 |
| nDCG@5 | 44.01% | 49.01% | +5.00 |
| Precision@10 | 6.00% | 6.25% | +0.25 |
| Recall@10 | 60.00% | 62.50% | +2.50 |
| nDCG@10 | 46.38% | 49.01% | +2.63 |

### 4. Architecture & Methodology Retrieval

**Dataset size:** 30 questions

| Metric | Baseline | Reranked | Δ (pp) |
|---|---:|---:|---:|
| MRR | 39.81% | 48.26% | +8.44 |
| Precision@1 | 30.00% | 36.67% | +6.67 |
| Recall@1 | 17.78% | 23.33% | +5.56 |
| nDCG@1 | 30.00% | 36.67% | +6.67 |
| Precision@2 | 18.33% | 23.33% | +5.00 |
| Recall@2 | 20.56% | 28.33% | +7.78 |
| nDCG@2 | 23.55% | 30.22% | +6.67 |
| Precision@3 | 17.78% | 20.00% | +2.22 |
| Recall@3 | 28.89% | 34.44% | +5.56 |
| nDCG@3 | 27.88% | 34.07% | +5.19 |
| Precision@5 | 13.33% | 16.67% | +3.33 |
| Recall@5 | 37.22% | 45.56% | +8.33 |
| nDCG@5 | 31.78% | 39.77% | +5.00 |
| Precision@10 | 9.67% | 11.67% | +2.00 |
| Recall@10 | 52.22% | 62.22% | +10.00 |
| nDCG@10 | 37.81% | 46.49% | +8.68 |

### Observations

- Reranking improved MRR across all four datasets.
- Multi-paper retrieval achieved the largest MRR improvement, increasing by **15.83 percentage points**.
- Single-paper retrieval achieved an MRR of **79.57%** with reranking.
- Numerical evidence-specific retrieval showed a smaller MRR gain of **2.39 percentage points**.
- Reranking does not improve every metric at every cutoff; multi-paper Recall@5 and Recall@10 decreased in the reranked setting.

*Answer-generation evaluation is not included in this report.*
