from src.german_credit_analysis.data_processing import DataProcessor
from src.german_credit_analysis.analysis import CreditAnalyzer
from src.german_credit_analysis.visualization import CreditVisualizer
from src.german_credit_analysis.modeling import CreditModeler
import matplotlib.pyplot as plt

# 1. loading and preprocessing data
print("Loading and preprocessing data...")
data_processor = DataProcessor('data/german_credit_data.csv')
data_processor.load_data()
data_processor.preprocess()
X, y = data_processor.get_data()
    
# 2. data analysis
print("\nAnalyzing data...")
analyzer = CreditAnalyzer(X, y)
print("Risk distribution:", analyzer.get_risk_distribution())
print("\nTop correlations with target:")
print(analyzer.get_correlation_with_target().head())
    
# 3. visualization
print("\nGenerating visualizations...")
visualizer = CreditVisualizer(X, y)
visualizer.plot_risk_distribution()
visualizer.plot_numerical_distributions()
visualizer.plot_correlation_heatmap()
    
# 4. modeling and evaluating
print("\nTraining and evaluating model...")
modeler = CreditModeler(X, y)
modeler.split_data()
modeler.train_model()
print("\nModel evaluation:")
print(modeler.evaluate_model()['classification_report'])
print("\nFeature importances:")
print(modeler.get_feature_importances().head())
    
 # show plots
plt.show()
