# Credit Risk Scoring with Missing Data Robustness & Human-in-the-Loop Prioritisation

## Overview
This project implements a robust machine learning pipeline for credit risk scoring using the Lending Club dataset (`accepted_2007_to_2018Q4.csv.gz`). The primary goal is to predict loan defaults while handling missing data effectively and introducing a "human-in-the-loop" review system for uncertain predictions.

## Methodology

### 1. Data Preprocessing & Cleaning
- **Target Variable Definition:** Loans are classified into binary outcomes: `0` for Good Loans ("Fully Paid") and `1` for Bad Loans ("Charged Off", "Default", etc.).
- **Feature Selection:** Selected 29 application-time features (e.g., loan amount, term, interest rate, employment length, credit history).
- **Missing Data Handling:** Explicit missing-data indicator columns were created for features with significant missing values (e.g., `mths_since_last_record`, `emp_length`). This helps the model learn patterns associated with the *absence* of data.
- **Feature Engineering:** Cleaned textual data (e.g., extracting numeric values from `term` and `emp_length`) and converted date columns into numeric features (e.g., `credit_history_years`).

### 2. Modeling
The dataset was split into 80% training and 20% testing sets using stratified sampling to maintain the default/non-default ratio. A Scikit-Learn `ColumnTransformer` was built to handle numerical (Median Imputation) and categorical (Most Frequent Imputation + One-Hot Encoding) features.

Three models were trained and evaluated, all utilizing class weighting (`class_weight="balanced"`) to handle the imbalanced nature of loan defaults:
1. **Logistic Regression (Baseline):** Basic pipeline without feature scaling.
2. **Improved Logistic Regression:** Enhanced pipeline incorporating `StandardScaler` for numerical features.
3. **Random Forest:** A non-linear ensemble approach (`n_estimators=100`, `max_depth=12`).

### 3. Human-in-the-Loop Review Prioritisation
To bridge the gap between automated decisions and human expertise, a prioritization system was developed based on prediction uncertainty. Applications are categorized into three groups based on the predicted probability of default:
- **High Risk - Manual Review:** Probability $\ge 0.70$
- **Low Risk - Auto Decision:** Probability $\le 0.30$
- **Uncertain - Human Review Priority:** $0.30 < \text{Probability} < 0.70$

An `uncertainty_score` is also calculated to quantify how close a probability is to the 0.50 decision boundary.

## Outputs & Generated Files
The script generates several analytical files evaluating the models and the prioritization logic:

### CSV Data Outputs
- **`model_comparison_results.csv`**: Contains performance metrics (Accuracy, Precision, Recall, F1 Score, ROC AUC) for all three models.
- **`human_review_prioritisation_results.csv`**: A detailed export of the test set predictions, containing actual outcomes, predicted probabilities, uncertainty scores, and the assigned review groups.

### Visualizations (PDFs)
- **`missing_data_analysis.pdf`**: A horizontal bar chart identifying the top features with missing values in the dataset.
- **`roc_curve_comparison.pdf`**: ROC curves comparing the Area Under the Curve (AUC) for the Logistic Regression, Improved Logistic Regression, and Random Forest models.
- **`model_comparison_figure.pdf`**: A grouped bar chart comparing the F1 Score and ROC AUC across the three evaluated models.
- **`confusion_matrix_best_model.pdf`**: A confusion matrix visualization for the best performing model (Improved Logistic Regression).
- **`human_review_prioritisation_figure.pdf`**: A bar chart displaying the distribution (counts) of loan applications across the three human-in-the-loop review categories.

## Conclusion
The inclusion of feature scaling significantly improved the baseline Logistic Regression model. The pipeline elegantly handles missing information by preserving its signal via missing-data indicators. Moreover, the human-in-the-loop review system allows for automated processing of clear-cut cases (High and Low Risk) while routing ambiguous applications to human underwriters, optimizing both efficiency and risk management.
