import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.freqs_ = {}

    def fit(self, X, y=None):
        for col in X.columns:
            freqs = X[col].value_counts(normalize=True)
            self.freqs_[col] = freqs
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col in X.columns:
            X_transformed[col] = X_transformed[col].map(self.freqs_[col]).fillna(0)
        return X_transformed

model = joblib.load("life_expectancy_model.pkl")  
X_test, y_test, y_pred = joblib.load("test_eval_data.pkl")
df = pd.read_csv("Life Expectancy Data.csv")
country_list = sorted(df["Country"].unique().tolist())

st.set_page_config(page_title="Life Expectancy Predictor", layout="centered")
st.title("Life Expectancy Prediction App")
st.markdown("""
This app predicts the **life expectancy** of a country based on health, economic, and social indicators.
Please fill in the inputs below and click **Predict**.
""")

with st.form("life_expectancy_form"):
    col1, col2 = st.columns(2)

    with col1:
        country = st.selectbox(
            "Country",
            country_list,
            help="Select the country you want to predict life expectancy for."
        )
        year = st.number_input(
            "Year",
            min_value=2000,
            max_value=2025,
            value=2015,
            help="Year for which prediction is being made."
        )
        status = st.selectbox(
            "Status",
            options=["Developed", "Developing"],
            help="Indicates whether the country is economically developed or developing."
        )
        adult_mortality = st.number_input(
            "Adult Mortality",
            help="Probability of dying between 15 and 60 years per 1000 adults."
        )
        infant_deaths = st.number_input(
            "Infant Deaths",
            help="Number of infant deaths (under 1 year) per 1000 live births."
        )
        percentage_expenditure = st.number_input(
            "% Expenditure",
            help="Expenditure on health as a percentage of GDP."
        )
        hepatitis_b = st.number_input(
            "Hepatitis B (%)",
            help="Immunization coverage for Hepatitis B among 1-year-olds (%)."
        )
        measles = st.number_input(
            "Measles Cases",
            help="Number of reported measles cases per 1000 population."
        )
        bmi = st.number_input(
            "BMI",
            help="Average Body Mass Index of the population (kg/m²)."
        )
        under_five_deaths = st.number_input(
            "Under-Five Deaths",
            help="Number of deaths of children under age 5 per 1000 live births."
        )

    with col2:
        polio = st.number_input(
            "Polio (%)",
            help="Immunization coverage against Polio among 1-year-olds (%)."
        )
        total_expenditure = st.number_input(
            "Total Expenditure (%)",
            help="Total health expenditure as a percentage of GDP."
        )
        diphtheria = st.number_input(
            "Diphtheria (%)",
            help="Immunization coverage for Diphtheria among 1-year-olds (%)."
        )
        hiv = st.number_input(
            "HIV/AIDS",
            help="Deaths due to HIV/AIDS per 1000 people."
        )
        gdp = st.number_input(
            "GDP",
            help="Gross Domestic Product per capita (in USD)."
        )
        population = st.number_input(
            "Population",
            help="Total population of the country."
        )
        thin_19 = st.number_input(
            "Thinness 1-19 years",
            help="Prevalence of thinness among children and adolescents (1–19 years)."
        )
        thin_5_9 = st.number_input(
            "Thinness 5-9 years",
            help="Prevalence of thinness among children aged 5–9 years."
        )
        income_comp = st.number_input(
            "Income Composition of Resources",
            help="Index combining income, education, and life expectancy (0 to 1)."
        )
        schooling = st.number_input(
            "Schooling (Years)",
            help="Average number of years of schooling a person receives."
        )


    submitted = st.form_submit_button("Predict")

if submitted:
    # Create DataFrame
    input_df = pd.DataFrame({
        'Country': [country],
        'Year': [year],
        'Status': [status],
        'Adult Mortality': [adult_mortality],
        'infant deaths': [infant_deaths],
        'percentage expenditure': [percentage_expenditure],
        'Hepatitis B': [hepatitis_b],
        'Measles': [measles],
        'BMI': [bmi],
        'under-five deaths': [under_five_deaths],
        'Polio': [polio],
        'Total expenditure': [total_expenditure],
        'Diphtheria': [diphtheria],
        'HIV/AIDS': [hiv],
        'GDP': [gdp],
        'Population': [population],
        'thinness  1-19 years': [thin_19],
        'thinness 5-9 years': [thin_5_9],
        'Income composition of resources': [income_comp],
        'Schooling': [schooling]
    })

    # Predict
    prediction = model.predict(input_df)
    predicted_life_expectancy = round(prediction[0], 2)

    st.success(f"Predicted Life Expectancy: **{predicted_life_expectancy}** years")

with st.expander("Model Performance Metrics (Click to expand)"):
    st.markdown("""
    - **Mean Squared Error (MSE):** `1.91`
    - **Mean Absolute Error (MAE):** `0.81`
    - **Root Mean Squared Error (RMSE):** `1.38`
    - **R² Score:** `0.978`
    - **Adjusted R² Score:** `0.977`
    """)
    st.info("These metrics indicate that the model performs with high accuracy and low error.")

with st.expander("What does each feature mean? Click to learn more!"):
    st.markdown("""
- **Country**: The nation you're exploring life expectancy for.  
- **Year**: The year of prediction (between 2000 and 2019).  
- **Status**: Whether the country is considered 'Developed' or 'Developing'.  

---

### Health & Mortality

- **Adult Mortality**: How many people (ages 15-60) die per 1000 individuals.
- **Infant Deaths**: Number of babies who died before turning 1 year old (per 1000 births).
- **Under-Five Deaths**: Children who died before age 5 (per 1000 births).

---

### Immunization

- **Hepatitis B (%)**: Percentage of 1-year-olds vaccinated for Hepatitis B.
- **Polio (%)**: Vaccination coverage against Polio among young children.
- **Diphtheria (%)**: Immunization rate for Diphtheria (serious throat infection).
- **Measles Cases**: Number of measles cases reported per 1000 people.

---

### Health and Body Stats

- **BMI**: Average Body Mass Index of the population — a measure of body fat.
- **Thinness 1-19 years**: Percent of thin people aged 1–19.
- **Thinness 5-9 years**: Percent of thin children aged 5–9.

---

### Economy & Lifestyle

- **% Expenditure**: How much of the country’s GDP goes to health.
- **Total Expenditure (%)**: Government + private spending on health (% of GDP).
- **GDP**: Gross Domestic Product — how wealthy the country is.
- **Income Composition of Resources**: A score combining income, education, and life expectancy (0–1 scale).
- **Schooling (Years)**: Average years of school per person.

---

### Other Risks

- **HIV/AIDS**: Deaths due to HIV/AIDS per 1000 people.

---
*These features help predict how long people might live in different parts of the world.*
    """)
with st.expander("Model Evaluation: Actual vs Predicted"):
    st.markdown("This chart shows how well the model predictions match the actual life expectancy values.")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel("Actual Life Expectancy")
    ax.set_ylabel("Predicted Life Expectancy")
    ax.set_title("Actual vs Predicted (Stacking Model)")

    st.pyplot(fig)
with st.expander("Model Evaluation: Distribution of Residuals"):
    residuals = y_test - y_pred

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(residuals, kde=True, ax=ax)
    ax.axvline(0, color='red', linestyle='--')
    ax.set_title("Distribution of Residuals (Prediction Errors)")
    ax.set_xlabel("Error")
    ax.set_ylabel("Frequency")
    
    st.pyplot(fig)