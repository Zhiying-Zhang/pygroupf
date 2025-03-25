from german_credit_analysis.data_processing import DataProcessor
from german_credit_analysis.analysis import DataAnalyzer
# from german_credit_analysis.visualization import DataVisualizer
# from german_credit_analysis.modeling import RiskPredictor

# 数据处理
processor = DataProcessor("data/german_credit_data.csv")
raw_data = processor.load_data()
clean_data = processor.clean_data()
encode_data = processor.encode_categorical_values() 
processed_data = processor.get_processed_data()

# 分析（添加风险列）
analyzer = DataAnalyzer(processed_data)
analyzer.add_risk_columns()
risk_report = analyzer.save_to_csv("data/risk_report.csv")

# print("Data processing and analysis completed successfully!")
# print("Risk report saved to: data/risk_report.csv")

# # 可视化
# visualizer = DataVisualizer(analyzed_data)
# visualizer.plot_risk_distribution()
# visualizer.plot_credit_vs_duration()

# # 机器学习
# predictor = RiskPredictor(analyzed_data)
# predictor.train_model()
