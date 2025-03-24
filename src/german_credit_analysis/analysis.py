import pandas as pd
from typing import Dict

class CreditAnalyzer:
    """Class for performing basic statistical analysis on German Credit data."""
    
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        """Initialize with feature matrix and target variable.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
        """
        self.X = X
        self.y = y
        self.data = X.copy()
        self.data['Risk'] = y
        
    def get_summary_stats(self) -> Dict:
        """Calculate summary statistics for numerical features.
        
        Returns:
            Dict: Dictionary containing summary statistics for each numerical feature.
        """
        numerical_cols = self.X.select_dtypes(include=['int64', 'float64']).columns
        assert len(numerical_cols) > 0, "No numerical columns found in data"
        
        stats = {}
        for col in numerical_cols:
            stats[col] = {
                'mean': self.X[col].mean(),
                'median': self.X[col].median(),
                'std': self.X[col].std(),
                'min': self.X[col].min(),
                'max': self.X[col].max()
            }
        return stats
    
    def get_risk_distribution(self) -> Dict:
        """Get distribution of risk classes.
        
        Returns:
            Dict: Count and percentage of each risk class.
        """
        counts = self.y.value_counts()
        percentages = self.y.value_counts(normalize=True) * 100
        
        return {
            'counts': counts.to_dict(),
            'percentages': percentages.round(2).to_dict()
        }
    
    def get_correlation_with_target(self) -> pd.Series:
        """Calculate correlation of numerical features with target variable.
        
        Returns:
            pd.Series: Correlation values sorted by absolute value in descending order.
        """
        numerical_cols = self.X.select_dtypes(include=['int64', 'float64']).columns
        assert len(numerical_cols) > 0, "No numerical columns found in data"
        
        corr = self.data[numerical_cols + ['Risk']].corr()['Risk'].drop('Risk')
        return corr.abs().sort_values(ascending=False)