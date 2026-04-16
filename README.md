# Stretch: Learning Curves Diagnostic

This repository contains the implementation of a learning curve diagnostic for a Logistic Regression model on a telecom churn dataset. This diagnostic tool helps identify whether the model is suffering from high bias (underfitting) or high variance (overfitting), guiding strategies for model improvement.

## Project Overview

The goal of this project is to use `sklearn.model_selection.learning_curve` to analyze model performance across different training set sizes. Given the class imbalance in the telecom dataset, we use the **F1-score** and **Stratified Cross-Validation** to ensure a robust diagnostic.

### Key Features
- **Data Preprocessing**: Handles numerical scaling and categorical one-hot encoding.
- **Learning Curve Plotting**: Generates a plot of training and validation scores with confidence bands.
- **Bias-Variance Analysis**: Provides a detailed written analysis based on the learning curve shapes.

## Getting Started

### Prerequisites
- Python 3.10+
- A virtual environment is recommended.

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Stretch-Learning-Curves-Diagnostic
   ```

2. **Set up a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the diagnostic script to generate the learning curve plot and see the score summary:

```bash
python learning_curve_diagnostic.py
```

This will:
1. Load and preprocess the `data/telecom_churn.csv` dataset.
2. Compute training and validation F1-scores across 5 different dataset sizes.
3. Save the resulting plot as `learning_curve.png`.
4. Output the final scores to the console.

## Project Structure

- `learning_curve_diagnostic.py`: Main Python script for data processing, model training, and plotting.
- `ANALYSIS.md`: Detailed interpretation of the learning curves and recommended next steps.
- `data/`: Directory containing the telecom churn dataset.
- `learning_curve.png`: The generated diagnostic plot.
- `requirements.txt`: List of required Python packages.
- `.gitignore`: Specifies files to be ignored by Git (e.g., `.venv`, `__pycache__`, images).

## Results & Analysis

The model was found to have **high bias**, indicating that it is too simple to capture the underlying patterns in the data. For a full breakdown of the diagnostic findings and recommendations, please refer to [ANALYSIS.md](ANALYSIS.md).
