import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, fbeta_score

# 读取数据
data = pd.read_csv("data/processed_credit_data.csv")

# 打印列名，检查是否与预期的列名相符
print(data.columns)

# 对分类特征进行 One-Hot Encoding
categorical_columns = ['sex', 'housing', 'saving_accounts', 'checking_account', 'purpose']  # 分类列名
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)  # drop_first=True 防止虚拟变量陷阱

# 创建目标列
data['good_credit'] = ((data['credit_amount'] > 10000) & (data['age'] > 30)).astype(int)

# 特征列 X 和目标列 y
X = data.drop(columns=['good_credit'])  # 特征列
y = data['good_credit']  # 目标列

# 切分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 设置超参数网格
param_grid = {
    "max_depth": [3, 5, 7, 10, None],
    "n_estimators": [3, 5, 10, 25, 50, 150],
    "max_features": [4, 7, 15, 20]
}

# 创建随机森林分类器
model = RandomForestClassifier(random_state=2)

# 使用网格搜索调整超参数
grid_search = GridSearchCV(model, param_grid=param_grid, cv=5, scoring='recall', verbose=4)
grid_search.fit(X_train, y_train)

# 打印最佳得分和最佳参数
print("Best score:", grid_search.best_score_)
print("Best params:", grid_search.best_params_)

# 使用最佳参数训练随机森林模型
rf = RandomForestClassifier(**grid_search.best_params_, random_state=2)
rf.fit(X_train, y_train)

# 预测测试集
y_pred = rf.predict(X_test)

# 输出评估指标
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("F2 Score:", fbeta_score(y_test, y_pred, beta=2))

