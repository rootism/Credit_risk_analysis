# Credit_risk_analysis


A machine learning project for predicting borrower credit risk using
demographic and loan-related attributes. The project compares multiple
classification algorithms and evaluates their performance before and
after addressing class imbalance.

## Project Overview

Credit risk modelling estimates the likelihood that a borrower will
default on their loan obligations. This project builds a classification
pipeline to predict `loan_status` and identify the model that performs
best for credit-risk prediction.

The project compares:

-   K-Nearest Neighbors (KNN)
-   Logistic Regression
-   Decision Tree
-   Random Forest

To improve data quality and model performance, the workflow includes
exploratory data analysis, categorical encoding, missing-value
imputation, outlier handling, feature scaling, and SMOTE-based class
balancing.

## Dataset

The dataset is sourced from Kaggle and contains **32,581 loan records**
with **12 features**.

Key features include:

-   `person_age` --- Borrower's age
-   `person_income` --- Annual income
-   `person_home_ownership` --- Home ownership status
-   `person_emp_length` --- Employment length
-   `loan_intent` --- Purpose of the loan
-   `loan_grade` --- Loan grade
-   `loan_amnt` --- Loan amount
-   `loan_int_rate` --- Loan interest rate
-   `loan_status` --- Target variable
-   `loan_percent_income` --- Loan amount as a percentage of income
-   `cb_person_default_on_file` --- Previous default indicator
-   `cb_person_cred_hist_length` --- Credit history length

## Methodology

### 1. Data Preprocessing

The dataset was inspected for duplicate records, missing values,
categorical variables, and outliers.

-   Checked and removed duplicate records during data cleaning.
-   Filled missing `person_emp_length` values using the mode.
-   Applied **KNN Imputation** to missing `loan_int_rate` values.
-   Encoded categorical variables using label encoding.
-   Removed extreme observations based on:
    -   Age \> 100
    -   Employment length \> 70 years
    -   Income \> 5 million
-   Applied **Min-Max Scaling** to normalize the features.
-   Split the data into **80% training and 20% testing** sets.

### 2. Exploratory Data Analysis

EDA was performed using boxplots, count plots, distributions, and a
Pearson correlation heatmap to examine relationships between borrower
characteristics, loan attributes, and loan status.

### 3. Class Imbalance

The original dataset contained:

-   Non-default loans: **25,467**
-   Default loans: **7,107**

**SMOTE (Synthetic Minority Over-sampling Technique)** was applied to
the training set, balancing both classes at **20,304 samples each**.

### 4. Model Training

Four classification models were trained and evaluated:

1.  KNN
2.  Logistic Regression
3.  Decision Tree
4.  Random Forest

Performance was evaluated using:

-   Accuracy
-   Precision
-   Recall
-   F1-score
-   Confusion Matrix

## Results

### Before SMOTE

  Model                   Accuracy   Weighted F1
  --------------------- ---------- -------------
  KNN                          90%          0.89
  Logistic Regression          85%          0.84
  Decision Tree                89%          0.89
  **Random Forest**        **93%**      **0.93**

### After SMOTE

  Model                   Accuracy   Weighted F1
  --------------------- ---------- -------------
  KNN                          82%          0.83
  Logistic Regression          78%          0.80
  Decision Tree                87%          0.87
  **Random Forest**        **93%**      **0.93**

### Best Model

**Random Forest** achieved the best overall performance with:

-   **93% accuracy**
-   **0.93 weighted F1-score**
-   **0.91 precision** for the positive/default class
-   **0.72 recall** for the positive/default class after SMOTE

The results indicate that Random Forest provided the strongest overall
predictive performance among the evaluated models.

## Tech Stack

-   Python
-   Pandas
-   NumPy
-   Matplotlib
-   Seaborn
-   Scikit-learn
-   Imbalanced-learn (SMOTE)
-   Google Colab / Jupyter Notebook

## Project Structure

``` text
Credit-Risk-Analysis/
│
├── MAIN.ipynb
├── credit_risk_dataset.csv
└── README.md
```

## How to Run

1.  Clone the repository.
2.  Open `MAIN.ipynb` in Jupyter Notebook or Google Colab.
3.  Install the required Python packages if needed:

``` bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn
```

4.  Upload `credit_risk_dataset.csv` when prompted.
5.  Run the notebook cells sequentially.

## Key Takeaways

-   Performed end-to-end credit-risk modelling from raw data
    preprocessing to model evaluation.
-   Used **KNN Imputation** for missing interest-rate values.
-   Applied **SMOTE** to address significant class imbalance.
-   Compared four machine learning classification algorithms.
-   **Random Forest** achieved the strongest overall performance with
    **93% test accuracy**.
