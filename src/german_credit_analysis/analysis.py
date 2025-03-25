import pandas as pd

class DataAnalyzer:
    def __init__(self, data):
        self.data = data

    def add_risk_column(self):
        # 根据业务逻辑定义风险（示例：金额 > 5000 且 Duration > 24 个月为高风险）
        self.data["risk"] = self.data.apply(
            lambda row: "high" if (row["Credit amount"] > 5000) and (row["Duration"] > 24) else "low",
            axis=1
        )
        return self.data

    def get_analyzed_data(self):
        return self.data