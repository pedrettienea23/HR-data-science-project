import pandas as pd
from datetime import datetime

def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Loads the HR dataset, performs base cleaning, and engineers all features
    required for both the attrition and performance models.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find the dataset at {filepath}")
    
    # Fix typos in Position
    df['Position'] = df['Position'].str.strip()
    
    # Drop records with missing ManagerID as they cannot be reliably imputed
    df = df.dropna(subset=['ManagerID']).copy()
    
    # Standardize datetime formats
    df['DateofHire'] = pd.to_datetime(df['DateofHire'])
    df['DateofTermination'] = pd.to_datetime(df['DateofTermination'])
    
    # Feature Engineering - Attrition Model
    # Calculate Tenure in Months
    today = datetime.now()
    df['Tenure_Months'] = (today - df['DateofHire']).dt.days / 30
    
    # Compensation Ratio (Salary vs Position Average)
    pos_avg_salary = df.groupby('Position')['Salary'].transform('mean')
    df['Comp_Ratio'] = df['Salary'] / pos_avg_salary
    
    # Engagement-Performance Gap
    df['Norm_Perf'] = df['PerfScoreID'] / 4
    df['Norm_Engage'] = df['EngagementSurvey'] / 5
    df['Engage_Perf_Gap'] = df['Norm_Perf'] - df['Norm_Engage']
    
    # Absence to Satisfaction Ratio (+1 to avoid division by zero)
    df['Absence_Satisfaction_Ratio'] = df['Absences'] / (df['EmpSatisfaction'] + 1)
    
    # Create binary target for high performers
    df['High_Performer'] = df['PerformanceScore'].isin(['Fully Meets', 'Exceeds']).astype(int)
    
    # Cast DeptID to string so it is treated as a categorical variable
    df['DeptID'] = df['DeptID'].astype(str)
    
    return df

if __name__ == "__main__":
    import os
    
    data_path = os.path.join('..', 'Data', 'HRDataset_v14.csv')
    
    print("Starting data preprocessing...")
    clean_df = load_and_clean_data(data_path)
    
    print(f"Data successfully cleaned and engineered.")
    print(f"Final shape: {clean_df.shape}")
    print("Ready for model training or inference!")