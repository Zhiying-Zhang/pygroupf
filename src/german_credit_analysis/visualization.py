import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Optional

class CreditVisualizer:
    """Class for visualizing German Credit data."""
    
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
        
    def plot_risk_distribution(self, figsize: tuple[int, int] = (6, 4)) -> plt.Figure:
        """Plot distribution of risk classes.
        
        Args:
            figsize (Tuple[int, int]): Size of the figure
            
        Returns:
            plt.Figure: Generated figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        sns.countplot(x='Risk', data=self.data, ax=ax)
        ax.set_title('Distribution of Credit Risk')
        ax.set_xlabel('Risk (0=Bad, 1=Good)')
        ax.set_ylabel('Count')
        return fig
    
    def plot_numerical_distributions(self, cols: Optional[List[str]] = None, 
                                   figsize: tuple[int, int] = (12, 8)) -> plt.Figure:
        """Plot distributions of numerical features by risk class.
        
        Args:
            cols (Optional[List[str]]): List of columns to plot. If None, plots all numerical columns.
            figsize (Tuple[int, int]): Size of the figure
            
        Returns:
            plt.Figure: Generated figure
        """
        numerical_cols = self.X.select_dtypes(include=['int64', 'float64']).columns
        if cols is not None:
            numerical_cols = [col for col in cols if col in numerical_cols]
        assert len(numerical_cols) > 0, "No numerical columns found in data"
        
        n_cols = 2
        n_rows = (len(numerical_cols) + 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()
        
        for i, col in enumerate(numerical_cols):
            sns.boxplot(x='Risk', y=col, data=self.data, ax=axes[i])
            axes[i].set_title(f'Distribution of {col} by Risk')
        
        # Hide any empty subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
            
        plt.tight_layout()
        return fig
    
    def plot_correlation_heatmap(self, figsize: tuple[int, int] = (10, 8)) -> plt.Figure:
        """Plot correlation heatmap for numerical features.
        
        Args:
            figsize (Tuple[int, int]): Size of the figure
            
        Returns:
            plt.Figure: Generated figure
        """
        numerical_cols = self.X.select_dtypes(include=['int64', 'float64']).columns
        assert len(numerical_cols) > 0, "No numerical columns found in data"
        
        corr = self.data[numerical_cols].corr()
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', 
                   center=0, ax=ax)
        ax.set_title('Correlation Heatmap of Numerical Features')
        return fig