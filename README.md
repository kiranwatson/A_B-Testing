# A/B Testing with H2O.ai's AutoML: README
## Overview:
This project explores the application of A/B testing techniques in evaluating the performance of different machine learning algorithms for predicting conversion rates in an e-commerce scenario. Using data
from two groups (control and test) derived from the datasets control_data.csv and experiment_data.csv, respectively, the project aims to identify the algorithm that best predicts user conversions.

## Project Structure:
### Data:
1. control_data.csv: Dataset containing metrics from the control group.
2. experiment_data.csv: Dataset containing metrics from the test group.
### Notebooks:
ab_testing.py: Jupyter Notebook containing the A/B testing process.In this .py file I have performed data preprocessing and build individual models like linear Regression, Decision tree , XgBoost . Utilized 
H2O.ai's AutoML to automate the model selection process.Trained multiple models and compared their performance using RMSE, R2_SCORE, MAE.Identified the best-performing algorithm based on chosen evaluation metrics.
### Notes:
1. Ensure that the datasets control_data.csv and experiment_data.csv are placed in the Data directory before running the notebooks.
2. Adjust the parameters and evaluation metrics as per project requirements.
3. Experiment with different algorithms and hyperparameters for model tuning and optimization.

