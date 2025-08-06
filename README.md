# Arabic-English Receipt Classification

A Data Sience project for classifying Arabic-English receipt items into categories using various embedding models and similarity-based approaches.

## Project Overview

This project focuses on classifying receipt items written in Arabic and English into 44 distinct categories based on item names. The classification system uses embedding models with cosine similarity to achieve accurate categorization.

## Current Performance

**Best Model Performance:**
- **Model:** `paraphrase-multilingual-mpnet-base-v2`
- **F1 Score:** 0.397
- **Categories:** 38 (reduced from original 44)
- **Improvement over baseline:** 0.373

## 📊 Experimental Results

### Initial Approach
| Method | F1 Score | Notes |
|--------|----------|-------|
| Direct prediction with phi-2.Q4_K_M.gguf | 0.0 | Failed to produce meaningful results |

### Embedding-Based Approaches

| Model/Technique | F1 Score | Dummy Score | Improvement | Notes |
|-----------------|----------|-------------|-------------|-------|
| LaBSE (no preprocessing) | 0.168 | 0.027 | 0.141 | Baseline embedding approach |
| LaBSE + normalization | 0.180 | 0.027 | 0.153 | Text normalization added |
| LaBSE + [CLS] token | 0.144 | 0.027 | 0.117 | Used [CLS] instead of mean pooling |
| LaBSE + Arabic noise removal | 0.200 | 0.027 | 0.173 | Improved preprocessing |
| all-MiniLM-L6-v2 | 0.211 | 0.024 | 0.187 | Faster processing (35K samples in 8-10s) |
| paraphrase-multilingual-MiniLM-L12-v2 | 0.326 | 0.024 | 0.302 | Multilingual support |
| + Category preprocessing (38 categories) | 0.366 | 0.024 | 0.342 | Reduced categories from 44 to 38 |
| paraphrase-multilingual-mpnet-base-v2 | **0.397** | 0.024 | **0.373** | **Current best model** |

## Key Improvements

1. **Text Normalization:** Improved F1 score from 0.168 to 0.180
2. **Arabic Noise Removal:** Enhanced preprocessing boosted performance to 0.200
3. **Model Selection:** Switching to multilingual models significantly improved results
4. **Category Optimization:** Reducing categories from 44 to 38 improved classification accuracy
5. **Model Scaling:** Larger models (mpnet-base-v2) provided better performance

## Next Steps

### Model Experimentation
- [ ] Test additional multilingual embedding models
- [ ] Benchmark models with competitive size-to-performance ratios
- [ ] Evaluate domain-specific models for receipt/retail data

### Other Techniques
- [ ] **Ensemble Learning:** Combine predictions from multiple models
- [ ] **Model Fine-tuning:** Train models on receipt-specific data

### Performance Targets
- [ ] **Primary Goal:** Achieve F1 score of 0.7+
- [ ] **Secondary Goal:** Maintain competitive inference speed
- [ ] **Optimization:** Balance accuracy vs computational efficiency

## 📈 Performance Metrics

- **Current F1 Score:** 0.397
- **Target F1 Score:** 0.7+
- **Processing Speed:** 35K samples in 8-10 seconds
- **Categories:** 38 receipt item categories
