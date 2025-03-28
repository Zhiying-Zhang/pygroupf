import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder

class DataVisualizer:
    def __init__(self, data_path):
        """
        Initialize the DataVisualizer class.
        
        Args:
            data_path (str): Path to the CSV file containing the data.
        """
        self.data = pd.read_csv(data_path)
        
        # Create 'image' directory if it doesn't exist
        if not os.path.exists('image'):
            os.makedirs('image')
    
    def plot_heatmap(self):
        """
        Plot a heatmap showing the correlation between all variables.
        
        Returns:
            matplotlib.figure.Figure: The heatmap object.
        """
        plt.figure(figsize=(12, 10))
        
        # Create a copy of the data for encoding (to avoid modifying the original data)
        df_encoded = self.data.copy()
        
        # Encode categorical variables
        cat_cols = ['sex', 'job', 'housing', 'saving_accounts', 'checking_account', 'purpose', 'risk_level']
        for col in cat_cols:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
        
        # Calculate the correlation matrix
        corr = df_encoded.corr()
        
        # Plot the heatmap
        heatmap = sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', 
                             center=0, linewidths=0.5, linecolor='black')
        
        plt.title('Feature Correlation Heatmap', fontsize=16)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('image/heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Heatmap saved to image/heatmap.png")
        
        return heatmap
    
    def plot_risk_distribution(self):
        """
        Plot the distribution of risk levels among customers.
    
        Returns:
        matplotlib.figure.Figure: The distribution plot object.
        """
        plt.figure(figsize=(10, 6))
    
        # Count the number of customers in each risk level
        risk_counts = self.data['risk_level'].value_counts().sort_index()
    
        # Plot the bar chart with updated parameters to avoid deprecation warning
        ax = sns.barplot(x=risk_counts.index, y=risk_counts.values, 
                    hue=risk_counts.index, palette='viridis', legend=False)
    
        # Add value labels on top of each bar
        for p in ax.patches:
            ax.annotate(f'{p.get_height()}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center', 
                   xytext=(0, 10), 
                   textcoords='offset points')
    
        plt.title('Distribution of Risk Levels', fontsize=16)
        plt.xlabel('Risk Level', fontsize=12)
        plt.ylabel('Number of Customers', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
    
        # Save the plot
        plt.savefig('image/risk_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
        print("Risk level distribution plot saved to image/risk_distribution.png")
    
        return ax
    
    def visualize_all(self):
        """Generate and save all visualization plots."""
        self.plot_heatmap()
        self.plot_risk_distribution()
        print("All visualizations have been generated and saved to the 'image' folder.")

# # Example usage
# visualizer = DataVisualizer('data/risk_report.csv')
# visualizer.visualize_all()