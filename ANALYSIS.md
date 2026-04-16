# Learning Curve Diagnostic Analysis

## Metric Selection
The **F1-score** was chosen as the evaluation metric for this diagnostic due to the significant class imbalance in the telecom churn dataset (approximately 83.7% non-churn vs. 16.3% churn). Accuracy would be misleadingly high even for a model that predicts "no churn" for every customer. The F1-score provides a balanced measure of precision and recall for the minority class (churners), which is critical for a business objective where both missing potential churners (low recall) and incorrectly flagging loyal customers (low precision) have associated costs.

## Diagnostic Findings

### 1. High Bias vs. High Variance
Based on the learning curve shape, the model suffers primarily from **high bias (underfitting)**. Both the training and validation F1-scores converge at a relatively low value (around 0.35–0.37). While the gap between the two curves is small—indicating low variance and good generalization—the fact that the training score itself is low and plateaus quickly suggests that the logistic regression model is too simple to capture the underlying complexity of the churn patterns in this dataset.

### 2. Impact of Collecting More Data
Collecting more data is **unlikely to significantly improve validation performance**. The learning curves show that both training and validation scores have already converged and plateaued after approximately 600–900 training examples. Since the curves are flat and the gap is narrow, adding more of the same type of data will not help the model learn more complex relationships that it is currently incapable of representing.

### 3. Increasing Model Complexity
Increasing model complexity would **likely help improve performance**. Because the model is limited by high bias, it would benefit from a more flexible hypothesis space. This could be achieved by adding polynomial features, interaction terms between variables (e.g., `tenure` * `monthly_charges`), or by switching to a more powerful non-linear model such as a Random Forest, Gradient Boosting Machine, or a Support Vector Machine with a non-linear kernel.

### 4. Recommended Next Steps
My primary recommendation is to **increase the model's capacity through feature engineering or by switching to a more complex model**. Specifically:
- **Feature Engineering**: Generate interaction features and polynomial terms to capture non-linear relationships.
- **Model Selection**: Transition from a linear Logistic Regression to an ensemble method like **Random Forest** or **XGBoost**, which can naturally handle non-linearities and feature interactions.
- **Hyperparameter Tuning**: If sticking with Logistic Regression, explore different regularization strengths ($C$ parameter) or try different solvers, though a more complex model architecture is likely to yield better results.
