import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from preprocessing import load_and_clean_data

def train_attrition_model(df: pd.DataFrame, save_path: str):
    """
    Trains and saves the Attrition Risk Detection model.
    """
    print("Training Attrition Model...")
    
    categorical_features = ['Department', 'RecruitmentSource', 'Sex']
    numerical_features = [
        'Tenure_Months', 'Comp_Ratio', 'Engage_Perf_Gap', 
        'Absence_Satisfaction_Ratio', 'SpecialProjectsCount', 'Absences'
    ]
    
    X = pd.get_dummies(df[categorical_features + numerical_features], drop_first=True)
    y = df['Termd']
    
    rf_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=0))
    ])
    
    rf_pipeline.fit(X, y)
    joblib.dump(rf_pipeline, save_path)
    print(f"Attrition Model saved to {save_path}")


def train_performance_model(df: pd.DataFrame, save_path: str):
    """
    Trains and saves the Performance Driver Analysis model.
    """
    print("Training Performance Model...")
    
    perf_features = [
        'Absences', 'EngagementSurvey', 'EmpSatisfaction', 'DeptID'
    ]
    
    X_perf = pd.get_dummies(df[perf_features], drop_first=True)
    y_perf = df['High_Performer']
    
    perf_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=0))
    ])
    
    perf_pipeline.fit(X_perf, y_perf)
    joblib.dump(perf_pipeline, save_path)
    print(f"Performance Model saved to {save_path}")


if __name__ == "__main__":
    data_path = os.path.join('..', 'Data', 'HRDataset_v14.csv')
    models_dir = os.path.join('..', 'Saved models')
    
    os.makedirs(models_dir, exist_ok=True)
    
    print("Loading and preparing data...")
    clean_df = load_and_clean_data(data_path)
    
    attrition_model_path = os.path.join(models_dir, 'attrition_model.pkl')
    train_attrition_model(clean_df, attrition_model_path)

    performance_model_path = os.path.join(models_dir, 'performance_driver_analysis.pkl')
    train_performance_model(clean_df, performance_model_path)
    
    print("All models successfully trained and exported!")