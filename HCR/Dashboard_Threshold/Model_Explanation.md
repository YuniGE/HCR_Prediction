# ESI Threshold Prediction Model Explained (Simple Version)

## The Big Picture 🎯

We're trying to answer one question: **How many citations will a paper need to be considered "highly cited" in the future?**

That's exactly what this prediction model does!

---

## What We Discovered 🔍

After analyzing 8 years of ESI data (49 releases, 22 research fields, ~12,000 data points), we found a simple but powerful pattern:

> **The older a paper gets, the more citations it accumulates, and the higher the threshold becomes.**

This relationship follows a mathematical curve called a **"Power Growth"** pattern:

```
Threshold ≈ a × (Paper Age)^b + c
```

### In Plain English:

| Paper Age | Approximate Threshold |
|-----------|----------------------|
| 1 year old | ~20 citations |
| 5 years old | ~100 citations |
| 10 years old | ~200 citations |

The growth slows down over time — like how a tree grows fast when young, then slows down as it matures.

---

## The Model We Used 🤖

We used **Gradient Boosting** — here's how it works in simple terms:

### Step-by-Step Process:

1. **Start with a simple guess** (like the average threshold)
2. **Look at the errors** (where was the guess wrong?)
3. **Build a small "fix"** to correct those errors
4. **Repeat 100 times**, each time fixing the remaining errors
5. **Combine all the fixes** into one smart prediction

### Analogy:
Think of it like having **100 experts** working together, where each one learns from the mistakes of the previous ones. The final prediction combines all their wisdom!

---

## What Information Does the Model Use? 📊

| Feature | Why It Matters | Importance |
|---------|----------------|------------|
| **Paper Age** | Older papers = more citations = higher threshold | **70%+** |
| **Research Field** | Some fields cite more (Medicine > Math) | **20%** |
| **Publication Year** | Which year the paper was published | **5%** |
| **Release Date** | When ESI publishes the threshold | **5%** |

### Key Insight:
**Paper Age alone explains over 70% of the threshold variation!** This makes the model surprisingly simple yet accurate.

---

## How Good Is the Model? ✅

| Metric | Score | What It Means |
|--------|-------|---------------|
| **R² Score** | **98%** | Model explains 98% of threshold variation |
| **Test RMSE** | ~8 | Average prediction error is about 8 citations |
| **Test MAE** | ~5 | Most predictions are within 5 citations of actual |

### What 98% R² Means:
If you imagine all the factors that determine a threshold, our model captures 98% of them. Only 2% is unexplained "noise."

---

## A Simple Example 📝

**Question:** What will be the threshold for Clinical Medicine papers published in 2020, when ESI releases data in January 2026?

**Model's Thinking Process:**

```
Step 1: Calculate Paper Age
        2026 - 2020 = 6 years old

Step 2: Identify Field Characteristics
        Clinical Medicine = High-citation field
        
Step 3: Apply Historical Pattern
        6-year-old papers in Clinical Medicine typically need ~130 citations

Step 4: Final Prediction
        Threshold ≈ 131 citations
```

---

## Why This Works 💡

The key insight is that citation patterns are **surprisingly predictable**:

### Three Reasons:

1. **Consistent Accumulation**
   - Papers accumulate citations at a relatively stable rate
   - The "rich get richer" — highly cited papers keep getting cited

2. **Stable Field Cultures**
   - Different fields have consistent citation practices
   - Clinical Medicine always cites more than Mathematics

3. **Fixed Window**
   - ESI uses a rolling 10-year window
   - This creates predictable patterns in threshold changes

### The Magic:
While individual paper citations are unpredictable, the **threshold** (top 1% cutoff) follows a very stable pattern because it's based on thousands of papers!

---

## Model Architecture 🏗️

```
┌─────────────────────────────────────────────────────────┐
│                    RAW DATA                              │
│         49 Excel files (2017-2025)                       │
└─────────────────────┬───────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────┐
│               DATA PREPROCESSING                         │
│  • Parse release dates from filenames                    │
│  • Handle merged cells                                   │
│  • Combine into single dataset (11,858 records)         │
└─────────────────────┬───────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────┐
│              FEATURE ENGINEERING                         │
│  • Paper_Age = Release_Year - Publication_Year          │
│  • Paper_Age_Months (more granular)                     │
│  • Days_Since_Start (time trend)                        │
│  • Field encoding (22 categories)                       │
└─────────────────────┬───────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────┐
│              MODEL TRAINING                              │
│  • Algorithm: Gradient Boosting Regressor               │
│  • Trees: 100                                           │
│  • Max Depth: 6                                         │
│  • Learning Rate: 0.1                                   │
│  • Train/Test Split: 80%/20% (time-based)              │
└─────────────────────┬───────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────┐
│               VALIDATION                                 │
│  • Test R²: 98%                                         │
│  • Cross-validation: Consistent across folds            │
└─────────────────────┬───────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────┐
│              PREDICTIONS                                 │
│  • Generate forecasts for January 2026                  │
│  • All 22 fields × 10 publication years                 │
└─────────────────────────────────────────────────────────┘
```

---

## Comparison with Other Models 📈

We tested multiple approaches:

| Model | Test R² | Notes |
|-------|---------|-------|
| Ridge Regression | 95% | Good baseline, linear |
| Elastic Net | 94% | Handles multicollinearity |
| Random Forest | 97% | Good, but slower |
| **Gradient Boosting** | **98%** | **Best performance** ✓ |

Gradient Boosting won because it:
- Captures non-linear relationships
- Handles feature interactions well
- Is robust to outliers

---

## Limitations & Caveats ⚠️

### What the Model Cannot Predict:

1. **Sudden Field Changes**
   - If a field suddenly becomes "hot" (like AI in 2023), historical patterns may not apply

2. **ESI Methodology Changes**
   - If Clarivate changes how they calculate thresholds

3. **Black Swan Events**
   - Pandemics, major discoveries, or policy changes that disrupt citation patterns

### Recommendation:
Use predictions as **estimates**, not guarantees. The model works best for near-term forecasts (1-2 years ahead).

---

## One-Sentence Summary

> **We trained a Gradient Boosting model on 8 years of ESI data, learning that paper age and research field are the key predictors of citation thresholds, achieving 98% accuracy.**

---

## Files Reference

| File | Description |
|------|-------------|
| `Threshold_prediction.ipynb` | Full model code with visualizations |
| `Threshold_Combined.xlsx` | Processed dataset |
| `threshold_prediction_model.pkl` | Saved model for reuse |
| `Predicted_Thresholds_Jan2026.xlsx` | Predictions output |
| `threshold_dashboard.html` | Interactive visualization |

---

## How to Use the Model

### For New Predictions:

```python
import pickle

# Load model
with open('threshold_prediction_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

# Create predictor
predictor = ThresholdPredictor(**model_data)

# Predict
threshold = predictor.predict(
    research_field='CLINICAL MEDICINE',
    publication_year=2023,
    release_date='20260115'
)
print(f"Predicted threshold: {threshold}")
```

---

*Model developed for ESI Threshold Analysis Project*
*Last updated: December 2024*


