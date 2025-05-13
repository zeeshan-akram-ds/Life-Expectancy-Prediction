# Life Expectancy Prediction using Advanced Regression Techniques

## Project Overview

This project focuses on predicting **Life Expectancy** of countries using a combination of **data cleaning**, **feature engineering**, and a wide range of **regression algorithms**, including ensemble models and model stacking. The project not only explores robust preprocessing and encoding strategies but also carefully evaluates each model to select the best-performing one based on multiple regression metrics.

---

## Problem Statement

The goal is to predict the **Life Expectancy** of a country using various health, demographic, and economic indicators such as GDP, schooling, BMI, HIV/AIDS rate, and more. The data is sourced from the World Health Organization and covers statistics for 193 countries over several years.

---

## Dataset Description

- Total countries: **193**
- Key features include: `GDP`, `Schooling`, `BMI`, `Status`, `HIV/AIDS`, `Alcohol`, `Income Composition of Resources`, etc.
- Target variable: `Life Expectancy`

---

## Data Exploration & Cleaning

1. **Initial Cleanup**
   - Dropped column: `Alcohol`
   - Cleaned and standardized column names

2. **Missing Value Handling**
   - Applied a combination of:
     - Median/Mean imputation
     - Group-wise Median/Mean
     - Linear Interpolation
     - KNN Imputer
   - Removed duplicate records

3. **Outlier Detection and Treatment**
   - Visualized using **boxplots** and **histograms**
   - Applied **Winsorization** for outlier capping
   - Used **RobustScaler** to reduce the impact of remaining outliers

4. **Skewness Handling**
   - Applied log transformation to highly skewed features
   - Visualized before/after distributions

---

## Exploratory Data Analysis (EDA)

Several visualizations were created for deep understanding of relationships:

- Life Expectancy distribution and trend over years
- Average Life Expectancy by:
  - **Country Status** (Developed/Developing)
  - **Top 15 Countries**
- Yearly trends of key health/economic indicators
- Scatterplots for:
  - **Schooling vs Life Expectancy**
  - **GDP vs Life Expectancy**
  - **Income Composition vs Life Expectancy**
  - **BMI vs Life Expectancy**
- HIV/AIDS rate over time by country status
- Pair plots and correlation heatmaps to reveal inter-feature dependencies

---

## Feature Engineering

- **Winsorization** on selected columns to limit extreme values
- **Encoding**:
  - Custom `FrequencyEncoder` (user-defined class) for high-cardinality categorical columns like `Country`
  - `OrdinalEncoder` for binary categorical column: `Status`
- Final numeric transformation using `RobustScaler` in a full pipeline
- Target variable (`Life Expectancy`) log-transformed using `TransformedTargetRegressor`

---

## Model Training & Evaluation

### Base Model
- **Ridge Regression**
  - Wrapped using `TransformedTargetRegressor` for log transformation
  - Good performance but room for improvement

### Model Benchmarking
Tested **10+ models** and stored results in a comparative DataFrame:

| Model              | R² Score |
|--------------------|----------|
| Extra Trees        | 0.9775   |
| Random Forest      | 0.9702   |
| XGBoost            | 0.9649   |
| Gradient Boosting  | 0.9517   |
| Bagging Regressor  | 0.9693   |
| Voting Regressor   | 0.9764   |
| Elastic Net        | Lower    |
| SVM                | Worst    |
| Others (Lasso, LR) | Lower    |

### Final Model - **Stacking Regressor**

- Base models:
  - Extra Trees
  - Random Forest
  - XGBoost
- Final Estimator: **Ridge**
- Log transformation applied using `TransformedTargetRegressor`

**Final Evaluation Metrics:**

- **MSE**: 1.919
- **MAE**: 0.824
- **RMSE**: 1.385
- **R²**: 0.9777
- **Adjusted R²**: 0.9769

**Train/Test R² Score:**

- Train: 0.9992
- Test: 0.9769  

---

## Model Evaluation Visualizations

- **Actual vs Predicted Scatter Plot**
  - Predicted values align closely with actual, forming a near-perfect diagonal line.

- **Residuals Distribution Plot**
  - Residuals centered around zero
  - Normally distributed without heavy skewness or outliers

These plots confirm that the model generalizes well and errors are minimal and evenly distributed.

---

## Hyperparameter Tuning

- Tried **Optuna** and manual hyperparameter tuning on:
  - Individual models (e.g., XGBoost, Gradient Boosting)
  - Stacked model as a whole
- Result: **No significant improvement** over the untuned stacking model

---

## Tools & Libraries Used

- **Python**
- **Pandas**, **NumPy**, **Matplotlib**, **Seaborn**
- **Scikit-learn**, **XGBoost**
- **Optuna** (for hyperparameter optimization)
- **Custom Encoders** for frequency encoding
- **Streamlit** (for deployment)  
---

## Conclusion

- Successfully predicted Life Expectancy with a high degree of accuracy
- Developed a **robust pipeline** combining preprocessing, custom encoders, scaling, and log transformation
- Final stacked model outperformed all individual and ensemble models
- Performed **extensive analysis**, **model comparison**, and **visual evaluation**

---
