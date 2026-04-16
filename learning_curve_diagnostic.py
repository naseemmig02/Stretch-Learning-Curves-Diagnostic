import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def load_and_preprocess_data(filepath):
    """Loads and preprocesses the telecom churn dataset."""
    df = pd.read_csv(filepath)
    
    # Drop customer_id as it's not a feature
    X = df.drop(['customer_id', 'churned'], axis=1)
    y = df['churned']
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()
    
    # Create a preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first'), categorical_cols)
        ])
    
    return X, y, preprocessor

def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5), scoring='f1'):
    """
    Generate a simple plot of the test and training learning curve.
    Adapted from sklearn documentation.
    """
    if axes is None:
        _, axes = plt.subplots(1, 1, figsize=(10, 6))

    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel(f"Score ({scoring})")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       scoring=scoring,
                       return_times=True)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curve
    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    axes.legend(loc="best")

    return plt

if __name__ == "__main__":
    # Load and preprocess
    filepath = 'data/telecom_churn.csv'
    X, y, preprocessor = load_and_preprocess_data(filepath)
    
    # Define the full model pipeline
    # Using LogisticRegression with liblinear solver for small/medium datasets
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'))
    ])
    
    # Setup StratifiedKFold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Define training sizes (at least 5)
    train_sizes = np.linspace(0.1, 1.0, 5)
    
    # Metric choice: F1-score is used to handle class imbalance (16% churners)
    # It balances precision and recall, which is crucial for identifying churners
    # without too many false alarms.
    scoring = 'f1'
    
    # Generate plot
    title = "Learning Curves (Logistic Regression)"
    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(model, X, y, cv=cv, n_jobs=-1,
                       train_sizes=train_sizes,
                       scoring=scoring,
                       return_times=True)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    
    print("Training sizes:", train_sizes)
    print("Training F1-scores:", train_scores_mean)
    print("Validation F1-scores:", test_scores_mean)
    
    print(f"Final training F1-score: {train_scores_mean[-1]:.4f}")
    print(f"Final validation F1-score: {test_scores_mean[-1]:.4f}")
    
    plt_obj = plot_learning_curve(model, title, X, y, cv=cv, 
                                  train_sizes=train_sizes, scoring=scoring)
    
    # Save the plot
    plt_obj.savefig('learning_curve.png')
    print("Learning curve plot saved as learning_curve.png")
    plt_obj.show()
