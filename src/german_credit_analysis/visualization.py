import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from matplotlib.ticker import PercentFormatter

class DataVisualizer:
    """
    A class for visualizing German credit data and risk analysis results.
    Generates and saves all plots to an 'images' folder.
    
    Attributes:
        data (pd.DataFrame): The processed DataFrame containing credit data and risk scores.
        output_dir (str): Directory to save generated images.
    """
    
    def __init__(self, data, output_dir="images"):
        """
        Initialize the DataVisualizer with processed data.
        
        Args:
            data: DataFrame containing processed credit data with risk scores.
            output_dir: Directory to save generated images (default: "images").
        """
        assert isinstance(data, pd.DataFrame), "data must be a DataFrame"
        self.data = data
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _save_plot(self, filename):
        """Helper method to save the current plot with standard formatting."""
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot to {save_path}")
    
    def plot_risk_distribution(self, figsize=(10, 6)):
        """
        Plot the distribution of risk levels in the dataset.
        Saves plot as 'risk_distribution.png'.
        """
        plt.figure(figsize=figsize)
        
        risk_counts = self.data['risk_level'].value_counts(normalize=True).sort_index()
        
        ax = sns.barplot(
            x=risk_counts.index,
            y=risk_counts.values,
            order=["Low risk", "Medium-low risk", "Medium-high risk", "High risk"],
            palette="viridis"
        )
        
        for p in ax.patches:
            height = p.get_height()
            ax.text(
                p.get_x() + p.get_width()/2.,
                height + 0.01,
                '{:.1%}'.format(height),
                ha="center"
            )
        
        plt.title('Distribution of Risk Levels', fontsize=14)
        plt.xlabel('Risk Level', fontsize=12)
        plt.ylabel('Percentage', fontsize=12)
        plt.xticks(rotation=45)
        self._save_plot("risk_distribution.png")
    
    def plot_credit_vs_duration(self, figsize=(10, 6)):
        """
        Plot credit amount vs duration, colored by risk level.
        Saves plot as 'credit_vs_duration.png'.
        """
        plt.figure(figsize=figsize)
        
        sns.scatterplot(
            data=self.data,
            x='duration',
            y='credit_amount',
            hue='risk_level',
            hue_order=["Low risk", "Medium-low risk", "Medium-high risk", "High risk"],
            palette="viridis",
            alpha=0.7
        )
        
        plt.title('Credit Amount vs Duration by Risk Level', fontsize=14)
        plt.xlabel('Duration (months)', fontsize=12)
        plt.ylabel('Credit Amount', fontsize=12)
        plt.legend(title='Risk Level', bbox_to_anchor=(1.05, 1), loc='upper left')
        self._save_plot("credit_vs_duration.png")
    
    def plot_age_risk_relationship(self, figsize=(10, 6)):
        """
        Plot the relationship between age and risk score.
        Saves plot as 'age_risk_relationship.png'.
        """
        plt.figure(figsize=figsize)
        
        sns.boxplot(
            data=self.data,
            x='risk_level',
            y='age',
            order=["Low risk", "Medium-low risk", "Medium-high risk", "High risk"],
            palette="viridis"
        )
        
        plt.title('Age Distribution by Risk Level', fontsize=14)
        plt.xlabel('Risk Level', fontsize=12)
        plt.ylabel('Age', fontsize=12)
        plt.xticks(rotation=45)
        self._save_plot("age_risk_relationship.png")
    
    def plot_purpose_distribution(self, figsize=(12, 6)):
        """
        Plot the distribution of loan purposes by risk level.
        Saves plot as 'purpose_distribution.png'.
        """
        plt.figure(figsize=figsize)
        
        purpose_counts = self.data.groupby(['purpose', 'risk_level']).size().unstack()
        purpose_counts = purpose_counts.div(purpose_counts.sum(axis=1), axis=0)
        
        purpose_counts.plot(
            kind='bar',
            stacked=True,
            color=['#440154', '#3b528b', '#21918c', '#5ec962'],
            figsize=figsize
        )
        
        plt.title('Loan Purpose Distribution by Risk Level', fontsize=14)
        plt.xlabel('Purpose', fontsize=12)
        plt.ylabel('Percentage', fontsize=12)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.xticks(rotation=45)
        plt.legend(title='Risk Level', bbox_to_anchor=(1.05, 1), loc='upper left')
        self._save_plot("purpose_distribution.png")
    
    def plot_categorical_feature_distribution(self, feature, figsize=(10, 6)):
        """
        Plot distribution of a categorical feature by risk level.
        Saves plot as '{feature}_distribution.png'.
        """
        plt.figure(figsize=figsize)
        
        feature_counts = self.data.groupby([feature, 'risk_level']).size().unstack()
        feature_counts = feature_counts.div(feature_counts.sum(axis=1), axis=0)
        
        feature_counts.plot(
            kind='bar',
            stacked=True,
            color=['#440154', '#3b528b', '#21918c', '#5ec962'],
            figsize=figsize
        )
        
        plt.title(f'{feature.capitalize()} Distribution by Risk Level', fontsize=14)
        plt.xlabel(feature.capitalize(), fontsize=12)
        plt.ylabel('Percentage', fontsize=12)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.xticks(rotation=45)
        plt.legend(title='Risk Level', bbox_to_anchor=(1.05, 1), loc='upper left')
        self._save_plot(f"{feature}_distribution.png")
    
    def plot_numerical_feature_distribution(self, feature, figsize=(10, 6)):
        """
        Plot distribution of a numerical feature by risk level.
        Saves plot as '{feature}_distribution.png'.
        """
        plt.figure(figsize=figsize)
        
        sns.violinplot(
            data=self.data,
            x='risk_level',
            y=feature,
            order=["Low risk", "Medium-low risk", "Medium-high risk", "High risk"],
            palette="viridis",
            inner="quartile"
        )
        
        plt.title(f'{feature.capitalize()} Distribution by Risk Level', fontsize=14)
        plt.xlabel('Risk Level', fontsize=12)
        plt.ylabel(feature.capitalize(), fontsize=12)
        plt.xticks(rotation=45)
        self._save_plot(f"{feature}_distribution.png")
    
    def plot_correlation_heatmap(self, figsize=(10, 8)):
        """
        Plot a heatmap of correlations between numerical features and risk score.
        Saves plot as 'correlation_heatmap.png'.
        """
        plt.figure(figsize=figsize)
        
        numerical_cols = ['age', 'job', 'credit_amount', 'duration', 'risk_score']
        corr_data = self.data[numerical_cols].corr()
        
        sns.heatmap(
            corr_data,
            annot=True,
            cmap='coolwarm',
            vmin=-1,
            vmax=1,
            fmt=".2f",
            linewidths=0.5
        )
        
        plt.title('Correlation Heatmap', fontsize=14)
        self._save_plot("correlation_heatmap.png")
    
    def plot_all_visualizations(self):
        """
        Generate and save all standard visualizations for the dataset.
        """
        self.plot_risk_distribution()
        self.plot_credit_vs_duration()
        self.plot_age_risk_relationship()
        self.plot_purpose_distribution()
        self.plot_correlation_heatmap()
        
        categorical_features = ['sex', 'housing', 'saving_accounts', 'checking_account']
        for feature in categorical_features:
            self.plot_categorical_feature_distribution(feature)
        
        numerical_features = ['age', 'job', 'credit_amount', 'duration']
        for feature in numerical_features:
            self.plot_numerical_feature_distribution(feature)