import os
import matplotlib.pyplot as plt
import seaborn as sns

class DataVisualizer:
    def __init__(self, data, output_dir="images"):
        self.data = data
        self.output_dir = output_dir
        # 自动创建文件夹
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_risk_distribution(self):
        plt.figure(figsize=(8, 6))
        sns.countplot(x="risk", data=self.data)
        plt.title("Risk Distribution")
        output_path = os.path.join(self.output_dir, "risk_distribution.png")
        plt.savefig(output_path)
        plt.close()

    def plot_credit_vs_duration(self):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x="Duration", y="Credit amount", hue="risk", data=self.data)
        plt.title("Credit Amount vs Duration by Risk")
        output_path = os.path.join(self.output_dir, "credit_vs_duration.png")
        plt.savefig(output_path)
        plt.close()