import pandas as pd
class DataAnalyzer:
    """
    Analyzes credit data to calculate risk scores and levels using configurable rules.
    
    Attributes:
        data (pd.DataFrame): Processed credit data with required fields.
        scoring_rules (dict): Field-specific scoring rules configuration.
        risk_levels (list): Thresholds for risk classification (threshold, level_name).
    """
    
    def __init__(self, data, scoring_rules, risk_levels):
        """
        Initialize analyzer with data and scoring configuration.
        
        Args:
            data: DataFrame containing processed credit data.
            scoring_rules: Dictionary mapping fields to their scoring rules.
            risk_levels: List of (threshold, level_name) tuples in descending order.
        """
        assert isinstance(data, pd.DataFrame), "data must be a DataFrame"
        assert isinstance(scoring_rules, dict), "scoring_rules must be a dict"
        assert len(risk_levels) > 0, "risk_levels cannot be empty"
        
        self.data = data
        self.scoring_rules = scoring_rules
        self.risk_levels = sorted(risk_levels, key=lambda x: x[0], reverse=True)
    
    def calculate_field_score(self, field_name, value):
        """
        Calculate score for a single field based on configured rules.
        
        Args:
            field_name: Field to score (must exist in scoring_rules).
            value: Field value to evaluate.
            
        Returns:
            int: Calculated score (0 if no rule matches).
        """
        rules = self.scoring_rules.get(field_name, {})
        
        # Handle numeric range rules (for continuous variables like age/amount)
        if isinstance(rules, list):
            for rule in rules:
                if 'condition' in rule and rule['condition'](value):
                    return rule['score']
                elif 'threshold' in rule and value > rule['threshold']:
                    return rule['score']
            return 0
        
        # Handle categorical fields with default/specific scoring
        elif isinstance(rules, dict) and 'default' in rules:
            if value in rules.get('specific', {}):
                return rules['specific'][value]
            try:
                return rules['default'](value)
            except (TypeError, KeyError):
                return rules.get('specific', {}).get(0, 0)
        
        # Handle simple value mappings (for discrete categories)
        elif isinstance(rules, dict):
            return rules.get(value, 0)
        
        return 0
    
    def calculate_risk_score(self, row):
        """
        Calculate total risk score by summing all field scores.
        
        Args:
            row: DataFrame row containing customer data.
            
        Returns:
            int: Total score clamped to 0-100 range.
        """
        assert not row.empty, "Row data cannot be empty"
        
        score = 0

        # Sum scores for all configured fields
        for field_name in self.scoring_rules:
            normalized_name = field_name.lower().replace(" ", "_")
            score += self.calculate_field_score(field_name, row[normalized_name])
        
        # Ensure final score stays within bounds
        return min(max(score, 0), 100)
    
    def determine_risk_level(self, score):
        """
        Classify score into risk level based on thresholds.
        
        Args:
            score: Calculated risk score (0-100).
            
        Returns:
            str: Risk level name or 'Unknown' if no match.
        """
        assert 0 <= score <= 100, "Score must be between 0-100"
        
        # Check thresholds in descending order (highest risk first)
        for threshold, level in self.risk_levels:
            if score >= threshold:
                return level
        return "Unknown"
    
    def generate_risk_report(self):
        """
        Generate report with customer IDs, risk scores and levels.
        
        Returns:
            pd.DataFrame: Report with original data plus risk analysis columns.
        """
        assert not self.data.empty, "Data cannot be empty"
        
        report_df = self.data

        # Add sequential customer ID based on DataFrame index
        report_df['customer_id'] = report_df.index + 1

        # Calculate risk metrics
        report_df['risk_score'] = report_df.apply(self.calculate_risk_score, axis=1)
        report_df['risk_level'] = report_df['risk_score'].apply(self.determine_risk_level)
        
        cols = ['customer_id'] + [col for col in report_df.columns if col != 'customer_id']
        return report_df[cols]
    
    def save_risk_report(self, output_path='risk_report.csv'):
        """
        Save risk report to CSV file.
        
        Args:
            output_path: File path to save report (default: 'risk_report.csv').
        """
        assert output_path.endswith('.csv'), "Output path must be a CSV file"
        
        report_df = self.generate_risk_report()
        report_df.to_csv(output_path, index=False)
        print(f"Risk report saved to {output_path}")