from pygroupf.data_processing import DataProcessor
from pygroupf.analysis import DataAnalyzer
from pygroupf.visualization import DataVisualizer

# Define column types
categorical_cols = [
    "Sex",
    "Housing",
    "Saving accounts",
    "Checking account",
    "Purpose",
]
numerical_cols = ["Age", "Job", "Credit amount", "Duration"]

"""
Encode specific categorical columns with numerical values as per requirements:
- sex: male=1, female=0
- housing: own=2, free=1, rent=0
- saving_accounts: unknown=0, little=1, moderate=2, quite rich=3, rich=4
- checking_account: unknown=0, little=1, moderate=2, rich=3
"""

# Define mapping of categorical values to numerical values
mapping = {
    "Sex": {"male": 1, "female": 0, "unknown": -1},
    "Housing": {"own": 2, "free": 1, "rent": 0, "unknown": -1},
    "Saving accounts": {
        "unknown": 0,
        "little": 1,
        "moderate": 2,
        "quite rich": 3,
        "rich": 4,
    },
    "Checking account": {"unknown": 0, "little": 1, "moderate": 2, "rich": 3},
}

# Define scoring rules for different fields
scoring_rules = {
    'Age': [
        {'condition': lambda x: x < 20 or x > 70, 'score': 15},
        {'condition': lambda x: 20 <= x < 25 or 60 < x <= 70, 'score': 10},
        {'condition': lambda x: 25 <= x < 30 or 50 < x <= 60, 'score': 5}
    ],
    'Sex': {'male': 2, 'female': 0, 'unknown': 1},
    'Job': {0: 15, 1: 10, 2: 5, 3: 1},
    'Housing': {0: 15, 1: 10, 2: 5},
    'Saving accounts': {
        'default': lambda x: (4 - x) * 3 if x > 0 else 10,
        'specific': {0: 10}
    },
    'Checking account': {
        'default': lambda x: (3 - x) * 4 if x > 0 else 10,
        'specific': {0: 10}
    },
    'Credit amount': [
        {'threshold': 8000, 'score': 15},
        {'threshold': 5000, 'score': 10},
        {'threshold': 2000, 'score': 5}
    ],
    'Duration': [
        {'threshold': 36, 'score': 15},
        {'threshold': 24, 'score': 10},
        {'threshold': 12, 'score': 5}
    ],
    'Purpose': {
        'business': 10,
        'education': 10,
        'unknown': 8,
        'car': 5,
        'furniture/equipment': 5,
        'radio/TV': 3,
        'domestic appliances': 3,
        'repairs': 3,
        'vacation/others': 3
    }
}

# Define risk levels based on total score
risk_levels = [
    (70, "High risk"),
    (50, "Medium-high risk"),
    (30, "Medium-low risk"),
    (0, "Low risk")
]


# data processing
processor = DataProcessor("data/german_credit_data.csv")
processor.load_data()
processor.clean_data(categorical_cols, numerical_cols)
processor.encode_categorical_values(mapping)
processed_data = processor.get_processed_data()


# data analysis and save risk report
analyzer = DataAnalyzer(processed_data, scoring_rules, risk_levels)
risk_report = analyzer.generate_risk_report() 
analyzer.save_risk_report("data/risk_report.csv")


# visualization
visualizer = DataVisualizer(risk_report)
visualizer.visualize_all()

