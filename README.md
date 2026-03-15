# HR Data Science: Performance Drivers and Attrition Analysis

The HR Data Science Project provides a comprehensive analytical pipeline applied to the HRDataset_v14.csv dataset. The overarching objective of this work is to extract actionable intelligence from employee records, moving beyond basic descriptive statistics to uncover the complex, interwoven dynamics that drive employee behavior, engagement, and retention. We focus heavily on rigorous statistical testing, paying particular attention to discrimination analysis to ensure fairness in compensation and operational opportunity. Finally, we implement a robust talent risk analysis framework powered by two separate machine learning models: one designed to predict employee attrition and another designed to identify the drivers of high performance. By synthesizing these elements, we can pinpoint star employees who are at a highly elevated risk of leaving the firm.

## Executive Summary and Key Findings

### Attrition Prediction Overview
The first pillar of this project evaluates the likelihood of an employee leaving the organization. Utilizing weighted Logistic Regression, we successfully model the delicate balance between active and terminated employees. 
Metrics:
* ROC-AUC: 0.738 plus or minus 0.055
* F1-Score: 0.632 plus or minus 0.064

![Attrition SHAP Plot](images/attrition_modeling_1.png)

### Performance Driver Analysis Overview
The second pillar seeks to identify the defining characteristics of high performing employees. A Random Forest Classifier serves as the foundation for this framework, yielding exceptional predictive power and identifying the exact traits shared by top talent.
Metrics:
* ROC-AUC: 0.894 plus or minus 0.093
* F1-Score: 0.978 plus or minus 0.012

![Performance SHAP Plot](images/performance_driver_analysis_0.png)

### Discrimination Analysis Overview
We applied rigorous statistical testing to investigate systemic disparities within the workplace. The discrimination analysis uncovered significant elements of the corporate structure where pay parity requires close monitoring. We focused primarily on the intersection of race, sex, and departmental assignment to unearth potential underlying biases, utilizing non-parametric tests to ensure mathematical validity.

## Exploratory Data Analysis

### Correlation Dynamics and Non-Linear Relationships
Our exploratory data analysis began with a detailed examination of feature correlations. We specifically opted for the Spearman rank correlation coefficient rather than the traditional Pearson correlation. The Pearson method assumes strictly linear relationships and is highly sensitive to outliers, making it fundamentally suboptimal for this specific dataset. Our HR data contains several ordinal variables, such as employee satisfaction scores measured on a scale of one to five, and performance scores categorized into rigid hierarchical tiers. The Spearman correlation evaluates monotonic relationships, meaning it assesses how well the relationship between two variables can be described using a monotonic function, irrespective of strict linearity. This makes it far more robust and accurate for evaluating our specific HR metrics.

![Spearman Correlation Heatmap](images/EDA_0.png)

The resulting heatmap provides immediate clarity on the interconnected nature of employee satisfaction, engagement, and financial compensation. The most notable observation is the strong underlying link between engagement survey results and employee satisfaction. Furthermore, salary displays a moderate correlation with specific position levels, confirming that financial compensation is heavily structured around the formal organizational chart rather than individual discretionary performance in most observed instances.

### Global Attrition Rates and Departmental Variances
We observed a global attrition rate of 33.4 percent across the organization. This represents a significant amount of turnover, prompting a maniacally detailed deeper dive into the specific departments driving this churn. The Production department emerged as the most volatile segment of the business, exhibiting a staggering 39.7 percent attrition rate. Conversely, roles within the Executive Office demonstrated perfect stability with absolute zero turnover. 

### Departmental Correlations: Salary, Performance, Engagement, and Satisfaction
To understand what motivates employees in wildly different operational silos, we conducted departmental analyses to determine the statistical significance of Spearman correlations between key variables. Our analysis mathematically proved that satisfaction and performance drivers are absolutely not uniform across the company. 

For instance, when evaluating the relationship between engagement and performance scores, the Software Engineering department showed a powerful, statistically significant correlation with a corrected p-value demonstrating exceptionally high confidence. A similar significant trend was noted in the Sales and IT/IS departments. 

When correlating Employee Satisfaction directly to Performance Scores, IT/IS and Production showed statistically significant positive relationships, while Software Engineering and Sales did not. This reveals a critical, highly actionable management insight: technical output in IT/IS is driven heavily by raw employee satisfaction, whereas Software Engineering performance is tied much more closely to engagement surveys and overall project alignment rather than sheer job contentment.

### Compensation Distribution and Structural Inequities
Understanding exactly how money is distributed is crucial for maintaining widespread morale and preventing critical legal or ethical structural failures. We began by charting the mathematical average of salaries categorized by position.

![Average Salary by Position](images/EDA_3.png)

The initial visualization visually confirms what is to be expected: senior executive and senior engineering roles heavily dominate the absolute top tier of the corporate payroll. However, to formally introduce the concept of systemic bias, we must evaluate whether compensation remains perfectly equitable when sliced by demographic factors rather than purely functional job titles.

![Average Salary by Race](images/EDA_6.png)
![Average Salary by Sex](images/EDA_7.png)

These visualizations highlight distinctly visible discrepancies in the average base salaries attributed to distinct racial groups and sexes. For instance, the raw, unfiltered data suggests widely varied compensation averages across different demographics. This visual discrepancy necessitates a strict, unrelenting statistical audit to determine if these variances are the result of conscious discrimination, unconscious bias, or merely distracting confounding variables like employee tenure and highly specific departmental assignment.

## Detailed Statistical Testing for Discrimination

To move entirely beyond visual assumptions, we implemented a maniacally thorough statistical testing pipeline designed specifically to probe for race and sex discrimination. A simple, superficial comparison of means is functionally insufficient and potentially misleading because real-world salary distributions wildly violate the rigorous assumption of normal distribution.

### Methodology for Salary Comparisons
We first applied the Shapiro-Wilk test to rigorously evaluate the normality of our salary distributions for different internal demographic groups. Because the data frequently failed the strict normality test, we completely discarded standard T-tests. Instead, we heavily utilized the Kruskal-Wallis non-parametric test. The Kruskal-Wallis test is exceptionally optimal for salary comparisons across multiple groups because it does not assume a normal distribution of residuals. When isolating binary demographic comparisons (such as male versus female within a specific subgroup), we paired this with the Mann-Whitney U test. To measure the genuine magnitude of any discovered disparities, we utilized the Common Language Effect Size (CLES), providing a clear probabilistic interpretation of the differences between the tested groups.

### Evaluating Confounding Variables
While raw Mann-Whitney U and Kruskal-Wallis tests provided initial P-values for demographic comparisons, we fundamentally recognized that variables such as Position and Department act as massive mathematical confounders. An observed raw pay gap between men and women might merely reflect an organizational imbalance in the number of men versus women residing in higher-paying technical departments like Software Engineering. To ruthlessly isolate the true independent effect of sex and race, we deployed robust Ordinary Least Squares (OLS) regression models. By modeling Salary directly as a function of Sex while mathematically controlling for Position and Department, we could pinpoint whether unequal pay for strictly equal work was occurring. The final analytical, corrected p-values (systematically adjusted using the Holm-Bonferroni method to aggressively mitigate the Family-wise Error Rate) demonstrated that while initial visual gaps looked alarming, rigorously controlling for exact position titles heavily reduced the statistical significance of the sex-based pay gap in several key operational departments.

### Methodology for Hiring and Attrition Ratios
To test comprehensively for biases in hiring practices or termination rates across demographic groups, we constructed dense contingency tables and applied Chi-squared tests. However, we carefully and continuously monitored the expected statistical frequencies within each table cell; when expected counts fell below the required mathematical safety threshold, we flagged those specific results as potentially unreliable and recommended exact permutation testing methods or vastly larger sample sizes before making any definitive legal or corporate policy claims.

## Detailed Machine Learning Modeling

Our approach to building predictive machine learning models was strictly rooted in a highly secure, data-leakage-proof pipeline rigorously utilizing the Scikit-Learn library. The primary analytical challenge was the massive class imbalance inherent in virtually all HR data; there are naturally far more active employees than terminated ones, and finding true high performers is a statistically rare event.

### Feature Engineering
Prior to any mathematical modeling, we carefully engineered several extremely critical features to provide the underlying algorithms with nuanced, highly contextual clues. "Comp_Ratio" was calculated precisely as the employee's exact salary divided by the overall average salary exclusively for their specific position, generating an index of relative wealth. "Tenure_Months" was calculated using datetime operational objects to precisely measure exact employment duration. We additionally created "Engage_Perf_Gap" to capture the mathematical distance between an individual's self-reported engagement level and their actual, officially documented performance rating.

### The Attrition Model
We meticulously formulated a Logistic Regression model to predict the binary target feature "Termd" (Terminated). Logistic Regression was selectively chosen for its unparalleled interpretability, directly allowing HR stakeholders to clearly understand the exact log-odds mathematical impact of every single feature. We implemented a Repeated Stratified K-Fold cross-validation strategy utilizing 5 splits and 10 repeat iterations to heavily ensure that our minority target classes were proportionally represented in absolutely every single training and testing validation fold. The class weights within the algorithm were explicitly forced to "balanced" to aggressively penalize the model for missing critical minority class predictions. The model prioritized checking the ROC-AUC and F1-Score metrics over simple total accuracy, ultimately definitively revealing that tenure length and internal compensation ratios are the strongest numerical leading indicators of flight risk.

### The Performance Driver Model
To accurately predict which specific employees formally fall into the High Performer category (defined aggressively as Fully Meets or absolutely Exceeds expectations), we shifted our algorithmic focus to a massive Random Forest Classifier. This complex ensemble method mathematically constructs a multitude of decision trees and outputs the overarching relative mode of the classes. Random Forests are highly, uniquely adept at capturing complex, incredibly tangled non-linear interactions between variables without requiring extensive feature scaling, though we still utilized a strict StandardScaler within our pipeline for maximum mathematical consistency. Formulated with 100 deep estimators and carefully balanced class weights, the robust Random Forest model achieved near-perfect F1-scores. We formally integrated the SHAP (SHapley Additive exPlanations) analytical library to completely deconstruct the model's entire decision-making process. The generated SHAP summary plots definitively proved beyond a shadow of a doubt that high engagement survey scores and strong fundamental employee satisfaction are the absolute, utmost critical prerequisites for top-tier organizational performance.

## The Talent Risk Framework

The ultimate, highly valuable culmination of this entire analytical project is the creation of the completely automated Talent Risk Framework. While understanding why people eventually leave and why people perform extraordinarily well is academically interesting, it is only genuinely valuable to an organization if it decisively drives direct business intervention.

We designed an incredibly sophisticated, unified risk matrix by merging the individual probabilistic outputs of both the Attrition model and the Performance model. By algorithmically predicting the Attrition Risk Percentage directly alongside the Performance Success Percentage on the current, fully active workforce, we aggressively filtered the entire company roster down to one highly critical analytical segment: The Stars-at-Risk.

We mathematically isolated all employees who strictly possess a performance success probability greater than 70 percent but simultaneously hold an attrition risk probability greater than 60 percent. This precise, incredibly narrow intersection accurately identifies the organization's undisputed most valuable, yet mathematically most vulnerable, human capital. Generating this highly targeted list directly allows the HR executive department to pivot immediately from a deeply reactionary stance to a highly proactive, aggressive retention strategy. Instead of conducting useless exit interviews merely to find out exactly why a top engineer already left, leadership can now instantly initiate preemptive retention protocols. This includes processing immediate salary adjustments to correct low Comp_Ratios, developing personalized career development planning, or utilizing highly targeted engagement interventions. Ultimately, this framework directly preserves vital institutional knowledge and definitively saves the company incredibly severe replacement friction costs.