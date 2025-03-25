class DataAnalyzer:
    """
    A class to analyze customer data and calculate risk scores based on various factors.
    """

    def __init__(self, data):
        """
        Initialize the DataAnalyzer with customer data.
        
        Args:
            data (pd.DataFrame): Input DataFrame containing customer data.
                               Expected to have columns processed by DataProcessor.
        
        Raises:
            ValueError: If input data is empty.
        """
        assert not data.empty, "Input DataFrame cannot be empty"
        self.data = data
    
    
    def calculate_risk_score(self, row):
        """
        Calculate a risk score for a single customer based on multiple factors.
        Higher scores indicate higher risk.
        
        Args:
            row (pd.Series): A row of customer data from the DataFrame
        
        Returns:
            int: Risk score for the customer between 0 and 100
        
        Note:
            Scoring factors include:
            - Age (young and old customers have higher risk)
            - Sex (male slightly higher risk)
            - Job type (unskilled and non-resident highest risk)
            - Housing situation
            - Savings/checking accounts
            - Credit amount
            - Loan duration
            - Loan purpose
        """
        score = 0
        
        # 1. Age (young and old customers have higher risk)
        if row['age'] < 20 or row['age'] > 70:
            score += 15
        elif row['age'] < 25 or row['age'] > 60:
            score += 10
        elif row['age'] < 30 or row['age'] > 50:
            score += 5
            
        # 2. Sex (male=1 has slightly higher risk)
        if row['sex'] == 1:  # male
            score += 2
        elif row['sex'] == 0:  # femal
            score += 0 
            
        # 3. Job (unskilled and non-resident have highest risk)
        if row['job'] == 0:    # unskilled and non-resident
            score += 15
        elif row['job'] == 1:  # unskilled and resident
            score += 10
        elif row['job'] == 2:  # skilled
            score += 5
        elif row['job'] == 3:  # highly skilled
            score += 1
            
        # 4. Housing situation (own have lowest risk, free medium and rent the highest)
        if row['housing'] == 0:  # rent
            score += 15
        elif row['housing'] == 1:  # free
            score += 10
        elif row['housing'] == 2:  # own
            score += 5  
        
        # 5. Savings accounts (more savings, lower risk)
        if row['saving_accounts'] == 0:  
            score += 10  
        else:
            score += (4 - row['saving_accounts']) * 3
        
        # 6. Checking accounts (more savings, lower risk)
        if row['checking_account'] == 0:  
            score += 10  
        else:
            score += (3 - row['checking_account']) * 4
        
        # 7. Credit amount (higher amount, higher risk)
        if row['credit_amount'] > 8000:
            score += 15
        elif row['credit_amount'] > 5000:
            score += 10
        elif row['credit_amount'] > 2000:
            score += 5
            
        # 8. Loan duration (longer duration, higher risk)
        if row['duration'] > 36:
            score += 15
        elif row['duration'] > 24:
            score += 10
        elif row['duration'] > 12:
            score += 5
            
        # 9. Loan purpose (business and education highest risk)
        purpose = row['purpose']
        if purpose in ['business', 'education']:
            score += 10
        elif purpose == 'unknown':
            score += 8  
        elif purpose in ['car', 'furniture/equipment']:
            score += 5
        elif purpose in ['radio/TV', 'domestic appliances', 'repairs', 'vacation/others']:
            score += 3
        
        # Ensure score is within 0-100 range
        score = min(max(score, 0), 100)
        
        assert 0 <= score <= 100, "Risk score must be between 0 and 100"
        return score
    
    def determine_risk_level(self, score):
        """
        Convert numerical risk score into categorical risk level.
        
        Args:
            score (int): Risk score between 0-100
        
        Returns:
            str: Risk level category
        """
        if score >= 70:
            return ("High risk")
        elif score >= 50:
            return ("Medium-high risk")
        elif score >= 30:
            return ("Medium-low risk")
        else:
            return ("Low risk")
    
    def add_risk_columns(self):
        """
        Add risk score and risk level columns to the DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame with added 'risk_score' and 'risk_level' columns
        
        Raises:
            KeyError: If required columns are missing from the DataFrame
        """
        self.data['risk_score'] = self.data.apply(self.calculate_risk_score, axis=1)
        self.data['risk_level'] = self.data['risk_score'].apply(self.determine_risk_level)
        return self.data
    
    def save_to_csv(self, output_path):
        """Save the DataFrame with risk analysis to a CSV file."""
        self.data.to_csv(output_path, index=False)
        return self.data

