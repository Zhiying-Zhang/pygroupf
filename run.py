from german_credit_analysis.data_processing import DataProcessor
from german_credit_analysis.analysis import DataAnalyzer
from german_credit_analysis.visualization import DataVisualizer
from german_credit_analysis.modeling import RiskPredictor

# 数据处理
processor = DataProcessor("data/german_credit_data.csv")
raw_data = processor.load_data()
clean_data = processor.clean_data()

# 分析（添加风险列）
analyzer = DataAnalyzer(clean_data)
analyzed_data = analyzer.add_risk_column()

# 可视化
visualizer = DataVisualizer(analyzed_data)
visualizer.plot_risk_distribution()
visualizer.plot_credit_vs_duration()

# 机器学习
predictor = RiskPredictor(analyzed_data)
predictor.train_model()
