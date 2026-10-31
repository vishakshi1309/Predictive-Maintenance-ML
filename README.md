# Predictive Maintenance Using Machine Learning

## Project Overview

Predictive maintenance uses data and machine learning to estimate when equipment may require maintenance. This project focuses on predicting the **Remaining Useful Life (RUL)** of aircraft engines using the NASA C-MAPSS FD001 dataset.

The aim is to explore engine sensor data, develop regression models, and evaluate their ability to estimate remaining engine life.

## Objectives

- Analyse aircraft engine sensor data.
- Perform data preprocessing and feature engineering.
- Calculate Remaining Useful Life (RUL).
- Build and compare multiple regression models.
- Evaluate model performance using standard regression metrics.

## Machine Learning Models

The following regression algorithms were implemented:

1. Linear Regression
2. Polynomial Regression
3. Decision Tree Regressor
4. Random Forest Regressor
5. Gradient Boosting Regressor

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Streamlit
- Plotly
- Joblib

## Dataset

**NASA C-MAPSS (FD001)**

The dataset contains simulated aircraft engine run-to-failure data, including operational settings and sensor measurements.

Dataset source: [NASA Prognostics Center of Excellence](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

## Evaluation Metrics

The models were evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)

## Project Structure

```text
Predictive-Maintenance-ML/
├── data/
├── charts/
├── main.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## How to Run

1. Clone this repository:

```bash
git clone https://github.com/YOUR-USERNAME/Predictive-Maintenance-ML.git
```

2. Install the required libraries:

```bash
pip install -r requirements.txt
```

3. Download the NASA C-MAPSS FD001 dataset and place the required files in the `data/` folder.

4. Run the machine learning script:

```bash
python main.py
```

5. Launch the dashboard:

```bash
streamlit run app.py
```

## Learning Outcomes

This project provided practical experience in exploratory data analysis, feature engineering, regression modelling, model evaluation, and applying machine learning to a real-world predictive maintenance problem.

## Future Improvements

- Explore additional predictive modelling techniques.
- Improve RUL prediction accuracy.
- Investigate explainable AI techniques for understanding model predictions.

---

**Developed as a Machine Learning Project.**
