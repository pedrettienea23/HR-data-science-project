import os
import joblib
import pandas as pd
from preprocessing import load_and_clean_data

def run_risk_reward_analysis(data_path: str, models_dir: str):
    """
    Loads models and HR data, calculates risk and performance probabilities,
    and identifies high-performing employees who are at risk of leaving.
    """
    attrition_clf = joblib.load(os.path.join(models_dir, 'attrition_model.pkl'))
    performance_clf = joblib.load(os.path.join(models_dir, 'performance_driver_analysis.pkl'))

    df = load_and_clean_data(data_path)

    # Setup Features & Targets
    attr_cat = ['Department', 'RecruitmentSource', 'Sex']
    attr_num = ['Tenure_Months', 'Comp_Ratio', 'Engage_Perf_Gap', 
                'Absence_Satisfaction_Ratio', 'SpecialProjectsCount', 'Absences']
    
    X_attr = pd.get_dummies(df[attr_cat + attr_num], drop_first=True)
    y_attr = df['Termd']

    perf_features = ['Absences', 'EngagementSurvey', 'EmpSatisfaction', 'DeptID']
    X_perf = pd.get_dummies(df[perf_features], drop_first=True)
    y_perf = df['High_Performer'] 

    attrition_clf.fit(X_attr, y_attr)
    performance_clf.fit(X_perf, y_perf)

    df['Attrition_Risk_%'] = attrition_clf.predict_proba(X_attr)[:, 1] * 100
    df['Performance_Success_%'] = performance_clf.predict_proba(X_perf)[:, 1] * 100
    
    stars_at_risk = df[(df['Performance_Success_%'] > 70) & (df['Attrition_Risk_%'] > 60)]
    print(f"{len(stars_at_risk)} employees may leave.\n")
    
    top_risks = stars_at_risk[['Employee_Name', 'Position', 'Attrition_Risk_%', 'Performance_Success_%']].sort_values(
        by='Attrition_Risk_%', 
        ascending=False
    )

    print("--- Top 10 Highest Risk ---")
    print(top_risks.head(10))
    print("\n--- Bottom 10 Highest Risk ---")
    print(top_risks.tail(10))


if __name__ == "__main__":
    data_path = os.path.join('..', 'Data', 'HRDataset_v14.csv')
    models_dir = os.path.join('..', 'Saved models')
    
    print("Starting Talent Risk and Reward Analysis...\n")
    run_risk_reward_analysis(data_path, models_dir)