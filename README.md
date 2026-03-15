# HR Data Science: Performance Drivers & Attrition Analysis

This project leverages the `HRDataset_v14.csv` to uncover the underlying dynamics of employee behavior. We perform comprehensive statistical testing—with a focus on **discrimination analysis**—and implement a **talent risk framework** utilizing two machine learning models to identify high-performing employees at risk of leaving.

## 🚀 Key Insights at a Glance

### Attrition Prediction
We modeled employee attrition using weighted Logistic Regression. The model achieves a balance between precision and recall, focusing on the features that lead to turnover.

| Metric | Value |
| :--- | :--- |
| **ROC-AUC** | 0.738 ± 0.055 |
| **F1-Score** | 0.632 ± 0.064 |

![Attrition SHAP Plot](images/attrition_modeling_1.png)
*SHAP values highlight Tenure, Compensation Ratio, and Absence-Satisfaction gaps as primary drivers.*

### Performance Driver Analysis
A Random Forest Classifier was trained to identify "High Performers." This model demonstrates high predictive power in identifying top talent.

| Metric | Value |
| :--- | :--- |
| **ROC-AUC** | 0.894 ± 0.093 |
| **F1-Score** | 0.978 ± 0.012 |

![Performance SHAP Plot](images/performance_driver_analysis_0.png)
*Engagement Survey scores and Employee Satisfaction are the most significant predictors of high performance.*

---

## 🔍 Exploratory Data Analysis (EDA)

### Correlation Dynamics
We utilized **Spearman Correlation** instead of Pearson to better capture non-linear relationships and account for ordinal variables such as Satisfaction levels and Performance scores.

![Spearman Correlation Heatmap](images/EDA_0.png)
*The heatmap reveals critical links: Salary shows moderate correlation with position levels, while Engagement and Satisfaction are tightly coupled, influencing both performance and stability.*

### Attrition & Departmental Trends
The global attrition rate stands at **33.4%**. Notably, the **Production** department faces the highest churn at nearly 40%, whereas the **Executive Office** remains stable.

We performed departmental deep-dives, analyzing the significance of correlations between salary, performance, and engagement. Significant variances were found in how pay interacts with performance across technical vs. non-technical roles.

### Salary & Equity Analysis
Salary distribution is heavily influenced by position, but secondary factors warrant closer inspection.

![Average Salary by Position](images/EDA_3.png)

To introduce the potential for systemic bias, we analyzed salary averages across **Race** and **Sex**.

![Salary by Race](images/EDA_6.png)
![Salary by Sex](images/EDA_7.png)

---

## ⚖️ Discrimination Analysis

A rigorous statistical audit was conducted to identify pay gaps or hiring biases.
- **Groups**: Breakdown by Race and Sex across departments.
- **Statistical Tests**: We used **Kruskal-Wallis** tests for salary comparisons (as data violated normality assumptions) and **Chi-squared** tests for attrition/hiring ratios.
- **Findings**: While some departments show parity, others exhibit statistically significant differences in compensation tiers that are not fully explained by tenure or performance.

---

## 🤖 Predictive Modeling Pipeline

### Goal & Feature Engineering
The primary goal was to move from descriptive stats to predictive intervention. We engineered:
- **Comp_Ratio**: Salary relative to position average.
- **Tenure_Months**: Time since hiring.
- **Engage_Perf_Gap**: Discrepancy between employee effort and output.

### The Pipeline
We implemented a robust Scikit-Learn pipeline:
1. **Preprocessing**: One-Hot Encoding for categorical data and `StandardScaler` for numeric features.
2. **Model Selection**: We compared **Logistic Regression** (baseline) against **Random Forest**. Random Forest proved superior for performance analysis due to its ability to capture complex feature interactions.
3. **Metric Rationale**: Accuracy was discarded due to imbalanced classes (fewer terminations than active employees). Instead, we optimized for **F1-Score** (balance) and **ROC-AUC** (discrimination power).

| Model | Task | ROC-AUC | F1-Score |
| :--- | :--- | :--- | :--- |
| Logistic Regression | Attrition | 0.74 | 0.63 |
| Random Forest | Performance | 0.89 | 0.98 |

---

## 💡 Conclusion & Talent Risk
By combining both models, we generated a **Star-at-Risk** list: employees with high performance probability but also high attrition risk scores. This enables HR to proactively intervene with retention bonuses or career growth conversations for the organization's most valuable assets.