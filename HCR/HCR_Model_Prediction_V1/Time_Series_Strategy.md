# Time Series Analysis Strategy for HCR Prediction 📈

## Overview
Moving from **Snapshot Analysis** (static) to **Time Series Analysis** (dynamic) allows us to predict **future** status based on **historical trends**. This is critical because HCR status depends on the *race* between a paper's citation growth and the rising threshold.

---

## 1. Paper-Level Time Series: "The Race Against Thresholds" 🏎️

For a single paper $i$ published at time $t_{pub}$, we track its state at each release date $t$.

### Key Metrics
1.  **Citation Velocity ($v$)**:
    How many citations does the paper gain per period?
    $$ v_t = \frac{Citations_t - Citations_{t-1}}{\Delta t} $$

2.  **Threshold Velocity ($v_{th}$)**:
    How fast is the bar rising for this paper's cohort?
    $$ v_{th,t} = \frac{Threshold_t - Threshold_{t-1}}{\Delta t} $$

3.  **Safety Margin Trend ($\Delta Gap$)**:
    Is the paper pulling away from the threshold, or is the threshold catching up?
    $$ \Delta Gap_t = v_t - v_{th,t} $$
    *   **If $\Delta Gap > 0$**: Safe. The paper is outrunning the threshold.
    *   **If $\Delta Gap < 0$**: Danger. The threshold is rising faster than the paper is gaining citations.

### Application: "At-Risk" Classification
Use time series to flag papers that are currently HCPs but are predicted to drop out:
*   **Input**: Last 6 bimonthly citation counts.
*   **Model**: ARIMA or Simple Exponential Smoothing.
*   **Output**: Predicted citation count at $t+1$. Compare with predicted threshold.

---

## 2. Researcher-Level Time Series: "Career Momentum" 🚀

Track the researcher's aggregate statistics over time to identify rising stars vs. declining impact.

### Constructing Panel Data
Transform your bimonthly files into a structure like this:

| Researcher | Date | HCP_Count | Total_Citations | New_HCPs | Lost_HCPs | Momentum_Score |
|------------|------|-----------|-----------------|----------|-----------|----------------|
| Prof. A | 2023-01 | 5 | 1000 | 1 | 0 | +1 |
| Prof. A | 2023-03 | 6 | 1200 | 1 | 0 | +1 |
| Prof. A | 2023-05 | 5 | 1250 | 0 | 1 | -1 |

### Key Features
1.  **Net HCP Flow**:
    $$ Flow_t = New\_HCPs_t - Lost\_HCPs_t $$
    *   Positive flow indicates rising influence.

2.  **HCP Volatility**:
    Standard deviation of HCP count over the last 12 months. High volatility implies instability (borderline papers).

3.  **Pipeline Strength (The "Iceberg" Metric)**:
    Count of papers in the "Gap Zone" (e.g., 80-99% of threshold) over time.
    *   *Rising Pipeline*: Future HCR candidate.
    *   *Empty Pipeline*: Risk of losing HCR status if current HCPs age out.

---

## 3. Modeling Techniques 🧠

### A. Sliding Window Classification (The Practical Approach)
Use data from $t-2, t-1, t$ to predict status at $t+1$.

**Features vector for Model:**
*   `Current_HCP_Count`
*   `HCP_Count_Change_6months`
*   `Avg_Citation_Velocity`
*   `Avg_Gap_Trend`

**Target:**
*   `Is_HCR_in_Next_Period` (Binary)

### B. Survival Analysis (The Statistical Approach)
Model the "Time to Drop" for an HCP.
*   "What is the probability that Paper X remains an HCP for another 12 months?"
*   Allows you to discount "weak" HCPs when aggregating researcher scores.

---

## 4. Implementation Steps 🛠️

### Step 1: Data Linkage
You need to link papers across the bimonthly Excel files.
*   **Key**: `Accession Number` (WOS:xxxx) is the best unique identifier.
*   Create a master database where each paper has a time-series trace of citations.

### Step 2: Feature Extraction Code (Conceptual)

```python
def calculate_momentum(paper_history):
    """
    Input: DataFrame with ['Date', 'Citations', 'Threshold'] sorted by Date
    """
    # Calculate velocities
    paper_history['cit_velocity'] = paper_history['Citations'].diff()
    paper_history['th_velocity'] = paper_history['Threshold'].diff()
    
    # Calculate trend
    paper_history['gap_trend'] = paper_history['cit_velocity'] - paper_history['th_velocity']
    
    # Predict next status
    # If (Current_Gap + Gap_Trend * 6_months) < 0 -> Predicted Drop
    return paper_history
```

### Step 3: Visualization (Spaghetti Plot)
Visualize the "Gap Trajectories" of a researcher's top 10 papers.
*   **X-axis**: Time
*   **Y-axis**: Normalized Gap ((Cites - Thresh) / Thresh)
*   **Lines**: Individual Papers
*   **Insight**: Are the lines pointing up (healthy) or down (decaying)?

---

## Summary
By using time series, you move from answering **"Is he an HCR?"** to **"Will he remain an HCR?"**. This allows for proactive identification of future stars and warning signs for declining influence.

