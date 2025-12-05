# Highly Cited Researcher (HCR) Prediction Model

## Problem Overview

### Goal
Predict whether a researcher should be classified as a **Highly Cited Researcher (HCR)** based on:
1. Their papers' citation counts
2. The threshold values for highly cited papers
3. The temporal dynamics of citations

### Data Available
| Data Type | Time Range | Frequency | Content |
|-----------|------------|-----------|---------|
| Threshold Data | 2017 Oct - 2025 Nov | Bimonthly (49 releases) | Citation thresholds for 22 fields × 11 years |
| HCR Paper Lists | 2017 Oct - 2025 Nov | Bimonthly | Papers, authors, citation counts |
| Sample Data | July 2024 | Single snapshot | Geosciences field (6,268 papers) |

---

## Key Challenge: Time Effect 🕐

### The Problem
- **Older papers have higher thresholds** (more time to accumulate citations)
- A 2014 paper needs ~228 citations; a 2024 paper needs only ~3 citations
- Raw citation counts are NOT comparable across publication years

### Solution: Citation Normalization

We introduce **Normalized Citation Score (NCS)**:

```
NCS = (Actual Citations) / (Threshold for that year)
```

| Interpretation | NCS Value |
|----------------|-----------|
| Highly Cited Paper | NCS ≥ 1.0 |
| Near-Threshold (Potential HCP) | 0.8 ≤ NCS < 1.0 |
| Above Average | 0.5 ≤ NCS < 0.8 |
| Below Average | NCS < 0.5 |

**Example:**
- Paper A (2014): 250 citations, threshold = 228 → NCS = 250/228 = 1.10 ✓ Highly Cited
- Paper B (2020): 80 citations, threshold = 99 → NCS = 80/99 = 0.81 → Near-threshold

---

## The Gap Concept (x Parameter) 📊

### Intuition
Papers close to the threshold have **potential** to become highly cited as citations accumulate over time.

### Mathematical Definition

Define **Gap Score**:
```
Gap = Threshold - Actual Citations
```

Define **Relative Gap**:
```
Relative Gap = Gap / Threshold
```

### Gap Categories

| Category | Condition | Interpretation |
|----------|-----------|----------------|
| **Already HCP** | Gap ≤ 0 | Paper is already highly cited |
| **High Potential** | 0 < Relative Gap ≤ 0.1 | Within 10% of threshold |
| **Medium Potential** | 0.1 < Relative Gap ≤ 0.2 | Within 20% of threshold |
| **Low Potential** | 0.2 < Relative Gap ≤ 0.3 | Within 30% of threshold |
| **Unlikely** | Relative Gap > 0.3 | Probably won't reach threshold |

### Time-Adjusted Gap (Critical Innovation)

Since papers continue accumulating citations, we estimate **future citations** based on:

```
Expected Future Citations = Current Citations × Growth Factor(remaining_months)
```

Where `Growth Factor` is learned from historical data:
- Papers typically grow 5-15% per bimonthly period (field-dependent)
- Growth rate decreases with paper age

---

## Feature Engineering 🔧

### Paper-Level Features
```python
# Basic features
'publication_year'           # Year paper was published
'current_citations'          # Citation count at snapshot time
'paper_age_months'           # Age of paper in months

# Threshold-related features
'threshold'                  # Required citations for HCP status
'ncs'                        # Normalized Citation Score
'gap'                        # Citations below threshold
'relative_gap'               # Gap as percentage of threshold
'is_hcp'                     # Binary: 1 if already HCP

# Potential features
'months_remaining'           # Months before paper exits 10-year window
'expected_growth'            # Predicted citation growth
'hcp_probability'            # Probability of becoming HCP
```

### Researcher-Level Features (Aggregated)
```python
# Paper counts
'total_papers'               # Total papers in the field
'hcp_count'                  # Number of highly cited papers
'potential_hcp_count'        # Papers within gap threshold

# Citation metrics
'total_citations'            # Sum of all citations
'avg_ncs'                    # Average normalized citation score
'max_ncs'                    # Highest NCS among papers
'median_ncs'                 # Median NCS

# Productivity metrics
'hcp_rate'                   # hcp_count / total_papers
'citation_concentration'     # % of citations from top papers

# Potential metrics
'expected_future_hcp'        # Predicted future HCP count
'hcp_momentum'               # Trend in HCP accumulation
```

---

## Model Architecture 🏗️

### Stage 1: Paper HCP Probability Model

Predicts probability of a paper becoming highly cited:

```
Input Features:
├── current_citations
├── paper_age_months
├── relative_gap
├── publication_year
├── field
└── citation_velocity (citations per month)

Output: P(HCP) ∈ [0, 1]
```

**Algorithm**: Gradient Boosting Classifier
- Handles non-linear relationships
- Captures interaction between features
- Provides probability outputs

### Stage 2: Researcher HCR Classification Model

Predicts whether researcher should be HCR:

```
Input Features:
├── confirmed_hcp_count
├── expected_hcp_count (from Stage 1)
├── total_papers
├── avg_ncs
├── hcp_rate
├── citation_concentration
└── field

Output: Binary (HCR or not HCR)
```

**Algorithm**: Ensemble of
1. Gradient Boosting Classifier
2. Random Forest Classifier  
3. Logistic Regression (for interpretability)

---

## Training Strategy 📚

### Time-Based Cross-Validation

```
Timeline: 2017 Oct ─────────────────────────────> 2025 Nov

Fold 1: Train [2017-2019] → Validate [2020 Q1-Q2]
Fold 2: Train [2017-2020] → Validate [2021 Q1-Q2]
Fold 3: Train [2017-2021] → Validate [2022 Q1-Q2]
Fold 4: Train [2017-2022] → Validate [2023 Q1-Q2]
Fold 5: Train [2017-2023] → Validate [2024 Q1-Q2]
Final:  Train [2017-2024] → Predict [2025+]
```

### Label Definition

**Ground Truth for HCR:**
We need to define what makes someone an HCR. Options:
1. **Top 1%** of researchers by HCP count in the field
2. **Minimum HCP threshold** (e.g., ≥ 3 HCPs in the field)
3. **Official Clarivate HCR list** (if available)

---

## Implementation Code 💻

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class HCRPredictor:
    """
    Highly Cited Researcher Prediction Model
    
    This model predicts whether a researcher should be classified as HCR
    based on their papers' citation patterns and threshold dynamics.
    """
    
    def __init__(self, gap_threshold=0.2):
        """
        Args:
            gap_threshold: Maximum relative gap to consider paper as "potential HCP"
                          Default 0.2 means papers within 20% of threshold
        """
        self.gap_threshold = gap_threshold
        self.paper_model = None
        self.researcher_model = None
        self.scaler = StandardScaler()
        self.field_growth_rates = {}  # Store field-specific growth rates
        
    def calculate_paper_features(self, papers_df, thresholds_df, snapshot_date):
        """
        Calculate features for each paper based on thresholds.
        
        Args:
            papers_df: DataFrame with columns [DOI, Authors, Publication Date, Times Cited]
            thresholds_df: DataFrame with columns [Year, Threshold-Highly Cited]
            snapshot_date: Date when the data was captured (YYYYMMDD format)
        
        Returns:
            DataFrame with paper-level features
        """
        # Parse snapshot date
        snapshot_year = int(str(snapshot_date)[:4])
        snapshot_month = int(str(snapshot_date)[4:6])
        
        # Merge papers with thresholds
        papers = papers_df.copy()
        papers['pub_year'] = papers['Publication Date'].astype(int)
        
        # Add threshold for each paper's publication year
        threshold_dict = dict(zip(thresholds_df['Year'], thresholds_df['Threshold-Highly Cited']))
        papers['threshold'] = papers['pub_year'].map(threshold_dict)
        
        # Calculate basic features
        papers['paper_age_months'] = (snapshot_year - papers['pub_year']) * 12 + snapshot_month
        papers['gap'] = papers['threshold'] - papers['Times Cited']
        papers['relative_gap'] = papers['gap'] / papers['threshold']
        papers['ncs'] = papers['Times Cited'] / papers['threshold']  # Normalized Citation Score
        
        # Binary: is already HCP?
        papers['is_hcp'] = (papers['Times Cited'] >= papers['threshold']).astype(int)
        
        # Potential HCP: within gap threshold
        papers['is_potential_hcp'] = (
            (papers['relative_gap'] > 0) & 
            (papers['relative_gap'] <= self.gap_threshold)
        ).astype(int)
        
        # Citation velocity (approximate)
        papers['citation_velocity'] = papers['Times Cited'] / papers['paper_age_months'].clip(lower=1)
        
        # Months remaining in 10-year window
        papers['months_remaining'] = 120 - papers['paper_age_months']  # 10 years = 120 months
        papers['months_remaining'] = papers['months_remaining'].clip(lower=0)
        
        return papers
    
    def estimate_hcp_probability(self, papers):
        """
        Estimate probability of each paper becoming HCP.
        
        For papers already HCP: probability = 1.0
        For papers with gap: use growth projection
        """
        papers = papers.copy()
        
        # Already HCP
        papers.loc[papers['is_hcp'] == 1, 'hcp_probability'] = 1.0
        
        # For non-HCP papers, estimate based on growth potential
        non_hcp = papers['is_hcp'] == 0
        
        # Simple growth model: 
        # P(HCP) = sigmoid(velocity * remaining_time - gap) / threshold
        # Higher velocity + more time + smaller gap = higher probability
        
        if non_hcp.any():
            # Expected additional citations = velocity * remaining_months
            papers.loc[non_hcp, 'expected_additional'] = (
                papers.loc[non_hcp, 'citation_velocity'] * 
                papers.loc[non_hcp, 'months_remaining'] * 0.5  # Discount factor
            )
            
            # Projected total citations
            papers.loc[non_hcp, 'projected_citations'] = (
                papers.loc[non_hcp, 'Times Cited'] + 
                papers.loc[non_hcp, 'expected_additional']
            )
            
            # Probability based on projected NCS
            papers.loc[non_hcp, 'projected_ncs'] = (
                papers.loc[non_hcp, 'projected_citations'] / 
                papers.loc[non_hcp, 'threshold']
            )
            
            # Convert to probability using sigmoid
            def sigmoid(x):
                return 1 / (1 + np.exp(-5 * (x - 1)))  # Centered at NCS=1
            
            papers.loc[non_hcp, 'hcp_probability'] = sigmoid(
                papers.loc[non_hcp, 'projected_ncs']
            )
        
        return papers
    
    def aggregate_researcher_features(self, papers):
        """
        Aggregate paper-level features to researcher level.
        
        Each author gets credit for papers they co-authored.
        """
        # Explode authors (each paper has multiple authors)
        researcher_papers = []
        
        for idx, row in papers.iterrows():
            if pd.isna(row['Authors']):
                continue
            authors = [a.strip() for a in str(row['Authors']).split(';')]
            for author in authors:
                researcher_papers.append({
                    'author': author,
                    'is_hcp': row['is_hcp'],
                    'is_potential_hcp': row['is_potential_hcp'],
                    'hcp_probability': row['hcp_probability'],
                    'ncs': row['ncs'],
                    'citations': row['Times Cited'],
                    'pub_year': row['pub_year']
                })
        
        rp_df = pd.DataFrame(researcher_papers)
        
        # Aggregate by researcher
        researcher_stats = rp_df.groupby('author').agg({
            'is_hcp': 'sum',  # Count of HCPs
            'is_potential_hcp': 'sum',  # Count of potential HCPs
            'hcp_probability': 'mean',  # Average HCP probability
            'ncs': ['mean', 'max', 'median'],  # NCS statistics
            'citations': ['sum', 'mean', 'max'],  # Citation statistics
            'pub_year': 'count'  # Total papers
        }).reset_index()
        
        # Flatten column names
        researcher_stats.columns = [
            'author', 'hcp_count', 'potential_hcp_count', 'avg_hcp_prob',
            'avg_ncs', 'max_ncs', 'median_ncs',
            'total_citations', 'avg_citations', 'max_citations',
            'total_papers'
        ]
        
        # Calculate derived features
        researcher_stats['hcp_rate'] = (
            researcher_stats['hcp_count'] / researcher_stats['total_papers']
        )
        
        # Expected HCP count (sum of probabilities)
        expected_hcp = rp_df.groupby('author')['hcp_probability'].sum().reset_index()
        expected_hcp.columns = ['author', 'expected_hcp_count']
        researcher_stats = researcher_stats.merge(expected_hcp, on='author')
        
        # Citation concentration (% from top papers)
        def top_paper_concentration(group):
            sorted_cites = group.sort_values(ascending=False)
            top_n = max(1, len(sorted_cites) // 5)  # Top 20%
            return sorted_cites.head(top_n).sum() / sorted_cites.sum() if sorted_cites.sum() > 0 else 0
        
        concentration = rp_df.groupby('author')['citations'].apply(top_paper_concentration).reset_index()
        concentration.columns = ['author', 'citation_concentration']
        researcher_stats = researcher_stats.merge(concentration, on='author')
        
        return researcher_stats
    
    def fit(self, X, y):
        """
        Fit the HCR classification model.
        
        Args:
            X: Researcher features DataFrame
            y: Binary labels (1 = HCR, 0 = not HCR)
        """
        # Select features
        feature_cols = [
            'hcp_count', 'potential_hcp_count', 'avg_hcp_prob',
            'avg_ncs', 'max_ncs', 'median_ncs',
            'total_citations', 'avg_citations', 'max_citations',
            'total_papers', 'hcp_rate', 'expected_hcp_count',
            'citation_concentration'
        ]
        
        X_features = X[feature_cols].fillna(0)
        X_scaled = self.scaler.fit_transform(X_features)
        
        # Train ensemble
        self.models = {
            'gb': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
            'rf': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
            'lr': LogisticRegression(max_iter=1000, random_state=42)
        }
        
        for name, model in self.models.items():
            model.fit(X_scaled, y)
        
        return self
    
    def predict(self, X):
        """
        Predict HCR status for researchers.
        
        Returns ensemble prediction (majority vote).
        """
        feature_cols = [
            'hcp_count', 'potential_hcp_count', 'avg_hcp_prob',
            'avg_ncs', 'max_ncs', 'median_ncs',
            'total_citations', 'avg_citations', 'max_citations',
            'total_papers', 'hcp_rate', 'expected_hcp_count',
            'citation_concentration'
        ]
        
        X_features = X[feature_cols].fillna(0)
        X_scaled = self.scaler.transform(X_features)
        
        # Get predictions from each model
        predictions = np.zeros((len(X), len(self.models)))
        for i, (name, model) in enumerate(self.models.items()):
            predictions[:, i] = model.predict(X_scaled)
        
        # Majority vote
        return (predictions.mean(axis=1) >= 0.5).astype(int)
    
    def predict_proba(self, X):
        """
        Predict probability of being HCR.
        
        Returns average probability from ensemble.
        """
        feature_cols = [
            'hcp_count', 'potential_hcp_count', 'avg_hcp_prob',
            'avg_ncs', 'max_ncs', 'median_ncs',
            'total_citations', 'avg_citations', 'max_citations',
            'total_papers', 'hcp_rate', 'expected_hcp_count',
            'citation_concentration'
        ]
        
        X_features = X[feature_cols].fillna(0)
        X_scaled = self.scaler.transform(X_features)
        
        # Get probabilities from each model
        probas = np.zeros((len(X), len(self.models)))
        for i, (name, model) in enumerate(self.models.items()):
            probas[:, i] = model.predict_proba(X_scaled)[:, 1]
        
        # Average probability
        return probas.mean(axis=1)


def load_and_process_geosciences_data(filepath, thresholds_df, snapshot_date=20240711):
    """
    Load and process the Geosciences HCP data.
    """
    # Load papers
    papers = pd.read_excel(filepath, skiprows=5, header=None)
    papers.columns = ['Accession Number', 'DOI', 'PMID', 'Article Name', 'Authors', 
                      'Source', 'Research Field', 'Times Cited', 'Countries', 
                      'Addresses', 'Institutions', 'Publication Date']
    
    # Clean data
    papers = papers[papers['Article Name'] != 'Article Name']
    papers['Times Cited'] = pd.to_numeric(papers['Times Cited'], errors='coerce')
    papers['Publication Date'] = pd.to_numeric(papers['Publication Date'], errors='coerce')
    papers = papers.dropna(subset=['Times Cited', 'Publication Date', 'Authors'])
    
    # Get Geosciences thresholds for July 2024
    geo_thresholds = thresholds_df[
        (thresholds_df['RESEARCH FIELDS'] == 'GEOSCIENCES') &
        (thresholds_df['Released date'] == snapshot_date)
    ][['Year', 'Threshold-Highly Cited']]
    
    return papers, geo_thresholds


def create_hcr_labels(researcher_stats, hcp_threshold=3):
    """
    Create HCR labels based on HCP count threshold.
    
    Clarivate typically requires researchers to have multiple HCPs
    to be considered for HCR status.
    """
    # Method 1: Fixed threshold (e.g., >= 3 HCPs)
    researcher_stats['is_hcr_fixed'] = (researcher_stats['hcp_count'] >= hcp_threshold).astype(int)
    
    # Method 2: Top percentile (e.g., top 1%)
    hcp_99 = researcher_stats['hcp_count'].quantile(0.99)
    researcher_stats['is_hcr_top1pct'] = (researcher_stats['hcp_count'] >= hcp_99).astype(int)
    
    # Method 3: Combined (HCP count + citation impact)
    # Top researchers by HCP count AND high average NCS
    researcher_stats['is_hcr_combined'] = (
        (researcher_stats['hcp_count'] >= hcp_threshold) &
        (researcher_stats['avg_ncs'] >= 1.0)
    ).astype(int)
    
    return researcher_stats


# Example usage
if __name__ == "__main__":
    # Load threshold data
    thresholds = pd.read_excel('Threshold_Combined.xlsx')
    
    # Load and process Geosciences data
    papers, geo_thresholds = load_and_process_geosciences_data(
        'Geosciences_HighlyCited.xlsx',
        thresholds,
        snapshot_date=20240711
    )
    
    # Initialize predictor
    predictor = HCRPredictor(gap_threshold=0.2)
    
    # Calculate paper features
    papers_with_features = predictor.calculate_paper_features(
        papers, geo_thresholds, snapshot_date=20240711
    )
    
    # Estimate HCP probabilities
    papers_with_probs = predictor.estimate_hcp_probability(papers_with_features)
    
    # Aggregate to researcher level
    researcher_stats = predictor.aggregate_researcher_features(papers_with_probs)
    
    # Create labels
    researcher_stats = create_hcr_labels(researcher_stats, hcp_threshold=3)
    
    # Display statistics
    print("Researcher Statistics Summary:")
    print(f"Total researchers: {len(researcher_stats)}")
    print(f"HCR candidates (>=3 HCPs): {researcher_stats['is_hcr_fixed'].sum()}")
    print(f"Top 1% by HCP count: {researcher_stats['is_hcr_top1pct'].sum()}")
    
    # Train model (if labels available)
    # predictor.fit(researcher_stats, researcher_stats['is_hcr_fixed'])
    # predictions = predictor.predict(researcher_stats)
```

---

## Key Insights for Time Handling ⏰

### Problem
Raw citation counts are misleading because:
- 2014 paper with 200 citations < threshold (228)
- 2024 paper with 5 citations > threshold (3)

### Solution: Normalized Citation Score (NCS)

```
NCS = Actual Citations / Threshold
```

This makes all papers comparable regardless of publication year.

### Why This Works

| Publication Year | Threshold | Citations | Raw Rank | NCS | NCS Rank |
|------------------|-----------|-----------|----------|-----|----------|
| 2014 | 228 | 250 | 3rd | 1.10 | 2nd |
| 2020 | 99 | 150 | 2nd | 1.52 | 1st |
| 2024 | 3 | 10 | 1st | 3.33 | N/A* |

*2024 papers are very new, their high NCS may not be meaningful.

### Additional Time Adjustments

1. **Age-weighted NCS**: Give more weight to older papers (proven track record)
2. **Growth-adjusted Gap**: Account for expected citation growth
3. **Window Position**: Papers near the end of 10-year window have less growth potential

---

## HCR Determination Logic 🏆

### Clarivate's Approach (Simplified)
1. Identify all Highly Cited Papers in a field
2. Count how many HCPs each researcher has
3. Researchers with exceptional HCP counts become HCR candidates
4. Final selection considers geographic/institutional diversity

### Our Model's Approach

```
Step 1: For each paper
├── Calculate NCS (time-normalized)
├── Determine if HCP (NCS >= 1.0)
├── Estimate HCP probability (for non-HCPs)
└── Identify "potential HCPs" (within gap threshold)

Step 2: For each researcher
├── Count confirmed HCPs
├── Calculate expected HCPs (sum of probabilities)
├── Compute citation quality metrics
└── Generate HCR probability score

Step 3: Classification
├── If HCP_count >= threshold → HCR
├── If HCP_count + potential_HCPs >= threshold → Likely HCR
└── Use ensemble model for borderline cases
```

---

## Gap Parameter (x) Optimization 📈

### Empirical Calibration

Using historical data, we can determine optimal gap threshold:

```python
# Find the gap value that best predicts future HCPs
gap_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

for gap in gap_values:
    # Papers within gap at time t
    potential_hcps = papers[(papers['relative_gap'] > 0) & 
                           (papers['relative_gap'] <= gap)]
    
    # Check how many became HCPs at time t+1
    conversion_rate = check_future_hcp_status(potential_hcps)
    print(f"Gap={gap:.0%}: Conversion rate = {conversion_rate:.2%}")
```

**Expected Results:**
- Gap = 5%: High conversion (>70%), but few papers
- Gap = 10%: Good conversion (~50-60%)
- Gap = 20%: Moderate conversion (~30-40%)
- Gap = 30%: Low conversion (~15-20%)

**Recommendation**: Use gap = 20% as default (balances quantity and quality)

---

## Model Validation Strategy 🧪

### Temporal Validation
1. Train on 2017-2022 data
2. Validate on 2023 data
3. Check if predicted HCRs match actual HCR list

### Cross-Field Validation
1. Train on 21 fields
2. Validate on held-out field
3. Check generalization across fields

### Metrics
- **Precision**: Of predicted HCRs, how many are actual HCRs?
- **Recall**: Of actual HCRs, how many did we identify?
- **F1 Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Overall discrimination ability

---

## Summary

| Component | Method | Key Parameters |
|-----------|--------|----------------|
| Time Normalization | NCS = Citations / Threshold | - |
| Gap Analysis | Relative Gap = (Threshold - Citations) / Threshold | gap_threshold = 0.2 |
| Paper Prediction | Gradient Boosting (probability output) | n_estimators = 100 |
| Researcher Aggregation | Sum HCPs + Expected HCPs | - |
| HCR Classification | Ensemble (GB + RF + LR) | threshold = 3 HCPs |

---

## Files Reference

| File | Description |
|------|-------------|
| `HCR_Prediction_Model.py` | Main model implementation |
| `Threshold_Combined.xlsx` | Historical threshold data |
| `Geosciences_HighlyCited.xlsx` | Sample HCP data for training |

---

*Model designed for ESI Highly Cited Researcher Analysis*
*Last updated: December 2024*

