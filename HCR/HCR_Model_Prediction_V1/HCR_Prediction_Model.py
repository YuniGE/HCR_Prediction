"""
Highly Cited Researcher (HCR) Prediction Model
==============================================

This module predicts whether a researcher should be classified as a 
Highly Cited Researcher based on their papers' citation patterns.

Key Concepts:
1. Normalized Citation Score (NCS) - handles time effect
2. Gap Analysis - identifies potential future HCPs
3. Ensemble Classification - robust HCR prediction

HCR Prediction Project
Date: December 2024
"""

import pandas as pd
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# PART 1: DATA LOADING AND PREPROCESSING
# ==============================================================================

def load_threshold_data(filepath='Threshold_Combined.xlsx'):
    """Load the combined threshold data."""
    df = pd.read_excel(filepath)
    print(f"Loaded threshold data: {len(df)} records")
    print(f"Fields: {df['RESEARCH FIELDS'].nunique()}")
    print(f"Date range: {df['Released date'].min()} to {df['Released date'].max()}")
    return df


def load_hcp_data(filepath, skip_rows=5):
    """
    Load Highly Cited Papers data from Excel file.
    
    Args:
        filepath: Path to HCP Excel file
        skip_rows: Number of header rows to skip
    
    Returns:
        DataFrame with cleaned HCP data
    """
    papers = pd.read_excel(filepath, skiprows=skip_rows, header=None)
    papers.columns = [
        'Accession Number', 'DOI', 'PMID', 'Article Name', 'Authors',
        'Source', 'Research Field', 'Times Cited', 'Countries',
        'Addresses', 'Institutions', 'Publication Date'
    ]
    
    # Remove duplicate header rows
    papers = papers[papers['Article Name'] != 'Article Name']
    
    # Convert to numeric
    papers['Times Cited'] = pd.to_numeric(papers['Times Cited'], errors='coerce')
    papers['Publication Date'] = pd.to_numeric(papers['Publication Date'], errors='coerce')
    
    # Drop invalid rows
    papers = papers.dropna(subset=['Times Cited', 'Publication Date', 'Authors'])
    
    print(f"Loaded {len(papers)} papers")
    print(f"Publication years: {papers['Publication Date'].min()} to {papers['Publication Date'].max()}")
    print(f"Citation range: {papers['Times Cited'].min()} to {papers['Times Cited'].max()}")
    
    return papers


def get_thresholds_for_snapshot(thresholds_df, field, snapshot_date):
    """
    Get thresholds for a specific field and snapshot date.
    
    Args:
        thresholds_df: Full threshold DataFrame
        field: Research field name (e.g., 'GEOSCIENCES')
        snapshot_date: Date in YYYYMMDD format (e.g., 20240711)
    
    Returns:
        Dictionary mapping year -> threshold
    """
    mask = (
        (thresholds_df['RESEARCH FIELDS'] == field) &
        (thresholds_df['Released date'] == snapshot_date)
    )
    field_thresholds = thresholds_df[mask]
    
    if len(field_thresholds) == 0:
        raise ValueError(f"No thresholds found for {field} at {snapshot_date}")
    
    return dict(zip(field_thresholds['Year'], field_thresholds['Threshold-Highly Cited']))


# ==============================================================================
# PART 2: FEATURE ENGINEERING
# ==============================================================================

def calculate_normalized_citation_score(papers_df, threshold_dict):
    """
    Calculate Normalized Citation Score (NCS) for each paper.
    
    NCS = Actual Citations / Threshold
    
    This normalizes across publication years, solving the time effect problem.
    """
    papers = papers_df.copy()
    papers['pub_year'] = papers['Publication Date'].astype(int)
    
    # Map threshold to each paper
    papers['threshold'] = papers['pub_year'].map(threshold_dict)
    
    # Handle papers with no threshold (outside window)
    papers = papers.dropna(subset=['threshold'])
    
    # Calculate NCS
    papers['ncs'] = papers['Times Cited'] / papers['threshold']
    
    return papers


def calculate_gap_features(papers_df, snapshot_date, gap_threshold=0.2):
    """
    Calculate gap-related features for identifying potential HCPs.
    
    Args:
        papers_df: DataFrame with papers and thresholds
        snapshot_date: Snapshot date in YYYYMMDD format
        gap_threshold: Maximum relative gap to consider "potential HCP"
    
    Returns:
        DataFrame with additional gap features
    """
    papers = papers_df.copy()
    
    # Parse snapshot date
    snapshot_year = int(str(snapshot_date)[:4])
    snapshot_month = int(str(snapshot_date)[4:6])
    
    # Calculate paper age in months
    papers['paper_age_months'] = (
        (snapshot_year - papers['pub_year']) * 12 + snapshot_month
    )
    
    # Calculate gap
    papers['gap'] = papers['threshold'] - papers['Times Cited']
    papers['relative_gap'] = papers['gap'] / papers['threshold']
    
    # Binary features
    papers['is_hcp'] = (papers['Times Cited'] >= papers['threshold']).astype(int)
    papers['is_potential_hcp'] = (
        (papers['relative_gap'] > 0) &
        (papers['relative_gap'] <= gap_threshold)
    ).astype(int)
    
    # Citation velocity (citations per month)
    papers['citation_velocity'] = papers['Times Cited'] / papers['paper_age_months'].clip(lower=1)
    
    # Months remaining in 10-year window
    papers['months_remaining'] = 120 - papers['paper_age_months']
    papers['months_remaining'] = papers['months_remaining'].clip(lower=0)
    
    return papers


def estimate_hcp_probability(papers_df):
    """
    Estimate probability of each paper becoming HCP.
    
    Uses citation velocity and remaining time to project future citations.
    """
    papers = papers_df.copy()
    
    # Already HCP -> probability = 1.0
    papers['hcp_probability'] = 0.0
    papers.loc[papers['is_hcp'] == 1, 'hcp_probability'] = 1.0
    
    # For non-HCP papers
    non_hcp_mask = papers['is_hcp'] == 0
    
    if non_hcp_mask.any():
        # Project future citations (with discount factor for uncertainty)
        papers.loc[non_hcp_mask, 'expected_additional'] = (
            papers.loc[non_hcp_mask, 'citation_velocity'] *
            papers.loc[non_hcp_mask, 'months_remaining'] * 0.5
        )
        
        # Projected NCS
        papers.loc[non_hcp_mask, 'projected_citations'] = (
            papers.loc[non_hcp_mask, 'Times Cited'] +
            papers.loc[non_hcp_mask, 'expected_additional']
        )
        papers.loc[non_hcp_mask, 'projected_ncs'] = (
            papers.loc[non_hcp_mask, 'projected_citations'] /
            papers.loc[non_hcp_mask, 'threshold']
        )
        
        # Convert to probability using sigmoid
        x = papers.loc[non_hcp_mask, 'projected_ncs'].fillna(0)
        papers.loc[non_hcp_mask, 'hcp_probability'] = 1 / (1 + np.exp(-5 * (x - 1)))
    
    return papers


# ==============================================================================
# PART 3: RESEARCHER AGGREGATION
# ==============================================================================

def aggregate_to_researcher_level(papers_df):
    """
    Aggregate paper-level features to researcher level.
    
    Each author gets attributed their papers (including co-authored ones).
    """
    # Explode authors - each paper row becomes multiple author rows
    researcher_records = []
    
    for idx, row in papers_df.iterrows():
        if pd.isna(row['Authors']):
            continue
        
        authors = [a.strip() for a in str(row['Authors']).split(';')]
        for author in authors:
            if author:  # Skip empty strings
                researcher_records.append({
                    'author': author,
                    'is_hcp': row['is_hcp'],
                    'is_potential_hcp': row['is_potential_hcp'],
                    'hcp_probability': row['hcp_probability'],
                    'ncs': row['ncs'],
                    'citations': row['Times Cited'],
                    'pub_year': row['pub_year'],
                    'threshold': row['threshold'],
                    'relative_gap': row['relative_gap']
                })
    
    rp_df = pd.DataFrame(researcher_records)
    print(f"Total author-paper records: {len(rp_df)}")
    
    # Aggregate statistics
    agg_funcs = {
        'is_hcp': 'sum',
        'is_potential_hcp': 'sum',
        'hcp_probability': ['mean', 'sum'],
        'ncs': ['mean', 'max', 'median', 'std'],
        'citations': ['sum', 'mean', 'max'],
        'pub_year': 'count'
    }
    
    researcher_stats = rp_df.groupby('author').agg(agg_funcs).reset_index()
    
    # Flatten column names
    researcher_stats.columns = [
        'author', 'hcp_count', 'potential_hcp_count',
        'avg_hcp_prob', 'expected_hcp_count',
        'avg_ncs', 'max_ncs', 'median_ncs', 'std_ncs',
        'total_citations', 'avg_citations', 'max_citations',
        'total_papers'
    ]
    
    # Fill NaN std with 0
    researcher_stats['std_ncs'] = researcher_stats['std_ncs'].fillna(0)
    
    # Derived features
    researcher_stats['hcp_rate'] = (
        researcher_stats['hcp_count'] / researcher_stats['total_papers']
    )
    
    # Citation concentration (approximation)
    researcher_stats['citation_concentration'] = (
        researcher_stats['max_citations'] / researcher_stats['total_citations'].clip(lower=1)
    )
    
    print(f"Total unique researchers: {len(researcher_stats)}")
    
    return researcher_stats


# ==============================================================================
# PART 4: HCR CLASSIFICATION MODEL
# ==============================================================================

class HCRClassifier:
    """
    Ensemble classifier for Highly Cited Researcher prediction.
    
    Uses multiple models and combines their predictions for robustness.
    """
    
    def __init__(self, hcp_threshold=3):
        """
        Args:
            hcp_threshold: Minimum HCPs to be considered HCR candidate
        """
        self.hcp_threshold = hcp_threshold
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_cols = [
            'hcp_count', 'potential_hcp_count', 'avg_hcp_prob', 'expected_hcp_count',
            'avg_ncs', 'max_ncs', 'median_ncs', 'std_ncs',
            'total_citations', 'avg_citations', 'max_citations',
            'total_papers', 'hcp_rate', 'citation_concentration'
        ]
    
    def create_labels(self, researcher_df, method='fixed'):
        """
        Create HCR labels for training.
        
        Args:
            researcher_df: Researcher statistics DataFrame
            method: 'fixed' (threshold-based) or 'percentile' (top N%)
        
        Returns:
            Binary labels array
        """
        if method == 'fixed':
            return (researcher_df['hcp_count'] >= self.hcp_threshold).astype(int)
        elif method == 'percentile':
            threshold = researcher_df['hcp_count'].quantile(0.99)
            return (researcher_df['hcp_count'] >= threshold).astype(int)
        elif method == 'combined':
            return (
                (researcher_df['hcp_count'] >= self.hcp_threshold) &
                (researcher_df['avg_ncs'] >= 1.0)
            ).astype(int)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def prepare_features(self, researcher_df, fit=False):
        """Prepare features for modeling."""
        X = researcher_df[self.feature_cols].fillna(0)
        
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def fit(self, researcher_df, labels=None, label_method='fixed'):
        """
        Fit the ensemble classifier.
        
        Args:
            researcher_df: Researcher statistics DataFrame
            labels: Optional pre-defined labels
            label_method: Method for creating labels if not provided
        """
        # Create labels if not provided
        if labels is None:
            labels = self.create_labels(researcher_df, method=label_method)
        
        # Prepare features
        X = self.prepare_features(researcher_df, fit=True)
        y = labels.values if hasattr(labels, 'values') else labels
        
        print(f"Training on {len(X)} samples")
        print(f"Positive class (HCR): {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
        
        # Initialize and train models
        self.models = {
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100, max_depth=5, random_state=42
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100, max_depth=5, random_state=42
            ),
            'logistic_regression': LogisticRegression(
                max_iter=1000, random_state=42
            )
        }
        
        for name, model in self.models.items():
            model.fit(X, y)
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
            print(f"  {name}: CV AUC = {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        
        # Store feature importance from gradient boosting
        self.feature_importance_ = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.models['gradient_boosting'].feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 5 Most Important Features:")
        print(self.feature_importance_.head())
        
        return self
    
    def predict(self, researcher_df):
        """Predict HCR status (binary)."""
        X = self.prepare_features(researcher_df, fit=False)
        
        # Get predictions from all models
        predictions = np.zeros((len(X), len(self.models)))
        for i, model in enumerate(self.models.values()):
            predictions[:, i] = model.predict(X)
        
        # Majority vote
        return (predictions.mean(axis=1) >= 0.5).astype(int)
    
    def predict_proba(self, researcher_df):
        """Predict probability of being HCR."""
        X = self.prepare_features(researcher_df, fit=False)
        
        # Get probabilities from all models
        probas = np.zeros((len(X), len(self.models)))
        for i, model in enumerate(self.models.values()):
            probas[:, i] = model.predict_proba(X)[:, 1]
        
        # Average probability
        return probas.mean(axis=1)
    
    def evaluate(self, researcher_df, labels):
        """Evaluate model performance."""
        y_pred = self.predict(researcher_df)
        y_proba = self.predict_proba(researcher_df)
        
        print("Classification Report:")
        print(classification_report(labels, y_pred, target_names=['Not HCR', 'HCR']))
        
        print(f"\nROC-AUC Score: {roc_auc_score(labels, y_proba):.3f}")
        
        return {
            'predictions': y_pred,
            'probabilities': y_proba,
            'auc': roc_auc_score(labels, y_proba)
        }


# ==============================================================================
# PART 5: MAIN PIPELINE
# ==============================================================================

def run_full_pipeline(
    hcp_filepath='Geosciences_HighlyCited.xlsx',
    threshold_filepath='Threshold_Combined.xlsx',
    field='GEOSCIENCES',
    snapshot_date=20240711,
    gap_threshold=0.2,
    hcp_threshold=3
):
    """
    Run the complete HCR prediction pipeline.
    
    Args:
        hcp_filepath: Path to HCP data file
        threshold_filepath: Path to threshold data file
        field: Research field name
        snapshot_date: Snapshot date in YYYYMMDD format
        gap_threshold: Gap threshold for potential HCPs (default 0.2 = 20%)
        hcp_threshold: Minimum HCPs for HCR status
    
    Returns:
        Dictionary with results and trained model
    """
    print("=" * 60)
    print("HCR PREDICTION PIPELINE")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n[1] Loading data...")
    thresholds = load_threshold_data(threshold_filepath)
    papers = load_hcp_data(hcp_filepath)
    
    # Step 2: Get thresholds for this field and snapshot
    print(f"\n[2] Getting thresholds for {field} at {snapshot_date}...")
    threshold_dict = get_thresholds_for_snapshot(thresholds, field, snapshot_date)
    print(f"Thresholds: {threshold_dict}")
    
    # Step 3: Calculate paper features
    print("\n[3] Calculating paper features...")
    papers = calculate_normalized_citation_score(papers, threshold_dict)
    papers = calculate_gap_features(papers, snapshot_date, gap_threshold)
    papers = estimate_hcp_probability(papers)
    
    print(f"\nPaper statistics:")
    print(f"  Total papers: {len(papers)}")
    print(f"  Already HCP: {papers['is_hcp'].sum()} ({papers['is_hcp'].mean()*100:.1f}%)")
    print(f"  Potential HCP (gap<={gap_threshold*100:.0f}%): {papers['is_potential_hcp'].sum()}")
    print(f"  Average NCS: {papers['ncs'].mean():.2f}")
    
    # Step 4: Aggregate to researcher level
    print("\n[4] Aggregating to researcher level...")
    researcher_stats = aggregate_to_researcher_level(papers)
    
    print(f"\nResearcher statistics:")
    print(f"  Total researchers: {len(researcher_stats)}")
    print(f"  Max HCPs: {researcher_stats['hcp_count'].max()}")
    print(f"  Avg HCPs: {researcher_stats['hcp_count'].mean():.2f}")
    print(f"  Researchers with >={hcp_threshold} HCPs: {(researcher_stats['hcp_count'] >= hcp_threshold).sum()}")
    
    # Step 5: Train classifier
    print("\n[5] Training HCR classifier...")
    classifier = HCRClassifier(hcp_threshold=hcp_threshold)
    labels = classifier.create_labels(researcher_stats, method='fixed')
    
    # Split data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        researcher_stats, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Fit on training data
    classifier.fit(X_train, y_train)
    
    # Evaluate on test data
    print("\n[6] Evaluating model...")
    results = classifier.evaluate(X_test, y_test)
    
    # Step 6: Generate predictions for all researchers
    print("\n[7] Generating predictions for all researchers...")
    researcher_stats['hcr_probability'] = classifier.predict_proba(researcher_stats)
    researcher_stats['hcr_predicted'] = classifier.predict(researcher_stats)
    
    # Top HCR candidates
    print("\nTop 20 HCR Candidates:")
    top_candidates = researcher_stats.nlargest(20, 'hcr_probability')[
        ['author', 'hcp_count', 'potential_hcp_count', 'total_papers', 
         'avg_ncs', 'hcr_probability']
    ]
    print(top_candidates.to_string())
    
    return {
        'papers': papers,
        'researcher_stats': researcher_stats,
        'classifier': classifier,
        'thresholds': threshold_dict,
        'evaluation': results
    }


# ==============================================================================
# PART 6: VISUALIZATION
# ==============================================================================

def plot_analysis_results(results, save_path=None):
    """Generate visualization plots for the analysis."""
    papers = results['papers']
    researchers = results['researcher_stats']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: NCS Distribution
    ax1 = axes[0, 0]
    papers['ncs'].clip(upper=5).hist(bins=50, ax=ax1, color='steelblue', edgecolor='white')
    ax1.axvline(x=1.0, color='red', linestyle='--', label='Threshold (NCS=1)')
    ax1.set_xlabel('Normalized Citation Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('NCS Distribution (Capped at 5)')
    ax1.legend()
    
    # Plot 2: Gap Distribution
    ax2 = axes[0, 1]
    non_hcp = papers[papers['is_hcp'] == 0]
    non_hcp['relative_gap'].clip(upper=1).hist(bins=50, ax=ax2, color='coral', edgecolor='white')
    ax2.axvline(x=0.2, color='green', linestyle='--', label='Gap Threshold (20%)')
    ax2.set_xlabel('Relative Gap (to Threshold)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Gap Distribution for Non-HCP Papers')
    ax2.legend()
    
    # Plot 3: HCP Count Distribution
    ax3 = axes[0, 2]
    researchers['hcp_count'].clip(upper=20).hist(bins=20, ax=ax3, color='seagreen', edgecolor='white')
    ax3.axvline(x=3, color='red', linestyle='--', label='HCR Threshold (3 HCPs)')
    ax3.set_xlabel('Number of Highly Cited Papers')
    ax3.set_ylabel('Number of Researchers')
    ax3.set_title('HCP Count per Researcher')
    ax3.legend()
    
    # Plot 4: HCR Probability Distribution
    ax4 = axes[1, 0]
    researchers['hcr_probability'].hist(bins=50, ax=ax4, color='purple', edgecolor='white')
    ax4.set_xlabel('HCR Probability')
    ax4.set_ylabel('Number of Researchers')
    ax4.set_title('HCR Probability Distribution')
    
    # Plot 5: HCP vs Total Papers
    ax5 = axes[1, 1]
    ax5.scatter(
        researchers['total_papers'].clip(upper=50),
        researchers['hcp_count'].clip(upper=30),
        alpha=0.3, c='steelblue'
    )
    ax5.set_xlabel('Total Papers (capped at 50)')
    ax5.set_ylabel('HCP Count (capped at 30)')
    ax5.set_title('HCP Count vs Total Papers')
    
    # Plot 6: Feature Importance
    ax6 = axes[1, 2]
    importance = results['classifier'].feature_importance_.head(10)
    ax6.barh(importance['feature'], importance['importance'], color='teal')
    ax6.set_xlabel('Importance')
    ax6.set_title('Top 10 Feature Importance')
    ax6.invert_yaxis()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()
    
    return fig


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    import os
    
    # Change to the correct directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Check if files exist
    hcp_file = '../Geosciences_HighlyCited.xlsx'
    threshold_file = '../Threshold_Combined.xlsx'
    
    if not os.path.exists(hcp_file):
        hcp_file = 'Geosciences_HighlyCited.xlsx'
    if not os.path.exists(threshold_file):
        threshold_file = 'Threshold_Combined.xlsx'
    
    # Run pipeline
    results = run_full_pipeline(
        hcp_filepath=hcp_file,
        threshold_filepath=threshold_file,
        field='GEOSCIENCES',
        snapshot_date=20240711,
        gap_threshold=0.2,
        hcp_threshold=3
    )
    
    # Save results
    results['researcher_stats'].to_excel('HCR_Predictions_Geosciences.xlsx', index=False)
    print("\nResults saved to HCR_Predictions_Geosciences.xlsx")
    
    # Generate plots
    try:
        plot_analysis_results(results, save_path='HCR_Analysis_Plots.png')
    except Exception as e:
        print(f"Plotting skipped: {e}")

