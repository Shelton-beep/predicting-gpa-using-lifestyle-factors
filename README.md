Predicting GPA Using Lifestyle Factors

This repository contains a machine learning project focused on predicting students' GPA based on lifestyle factors like study hours, sleep, extracurricular activities, and stress levels.

Table of Contents

- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
  - [Clone the Repository](#clone-the-repository)
  - [Run the Notebook](#run-the-notebook)
- [Project Workflow](#project-workflow)
  - [1. Data Loading and Preprocessing](#1-data-loading-and-preprocessing)
  - [2. Exploratory Data Analysis (EDA)](#2-exploratory-data-analysis-eda)
  - [3. Model Training and Evaluation](#3-model-training-and-evaluation)
  - [4. Hyperparameter Tuning](#4-hyperparameter-tuning)
  - [5. Model Interpretation with SHAP](#5-model-interpretation-with-shap)
- [Results](#results)
  - [Insights](#insights)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Project Overview

The aim of this project is to examine the relationship between lifestyle choices and academic performance, using students' GPA as a measurable outcome. By building and evaluating predictive models, we hope to uncover which lifestyle factors most significantly influence GPA. Such insights can be valuable for educational institutions, advisors, and students themselves in making informed decisions to optimize academic performance.

## Dataset Description

The dataset comprises several lifestyle factors and the GPA of each student. Key features include:

- Study Hours Per Day: Average hours spent studying daily.
- Extracurricular Hours Per Day: Time spent on extracurricular activities (sports, clubs, etc.).
- Sleep Hours Per Day: Average hours of sleep per night.
- Social Hours Per Day: Time spent socializing with friends or family.
- Physical Activity Hours Per Day: Hours spent on physical activities (exercise, sports).
- Stress Level: Self-reported stress level (Low, Moderate, High), converted to numerical format for modeling.
- GPA: The target variable, representing the students' GPA.

## Project Structure

1. Data Loading and Exploration:

   - Load and inspect the dataset for missing values and data types.
   - Summarize descriptive statistics to understand the distribution and central tendencies of each variable.

2. Data Preprocessing:

   - Missing Values Handling: Impute missing values with the mean.
   - Encoding: Convert categorical variables like stress levels to numerical values.
   - Feature Scaling: Standardize features to ensure they are on a similar scale.
   - Polynomial Features: Generate polynomial terms to capture potential non-linear relationships.

3. Exploratory Data Analysis (EDA):

   - Visualize the distribution of GPA.
   - Examine correlations among variables to identify potential predictors.
   - Create histograms, scatter plots, and a heatmap to reveal relationships between lifestyle factors and GPA.

4. Model Training and Evaluation:

   - Train multiple regression models to predict GPA based on lifestyle features.
   - Models include Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting, and XGBoost.
   - Evaluate models using cross-validation and metrics such as R2, Mean Absolute Error (MAE), and Mean Squared Error (MSE).

5. Hyperparameter Tuning:

   - Fine-tune the hyperparameters of the best-performing model to enhance accuracy.

6. Model Interpretation with SHAP:
   - Use SHAP values to explain the importance of each feature, helping interpret the model's predictions.

## Prerequisites

Ensure you have Python installed, along with the following libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap
```

## Getting Started

## Clone the Repository

## Clone this repository to your local machine to get started:

```bash
git clone https://github.com/Shelton-beep/predicting-gpa-using-lifestyle-factors.git
cd predicting-gpa-using-lifestyle-factors
```

## Run the Notebook

To run the project, open and execute each cell in the `predicting-gpa-using-lifestyle-factors.ipynb` notebook. It contains detailed code and explanations of each step.

---

## Project Workflow

## 1. Data Loading and Preprocessing

In the first step, we load the dataset and perform basic data preprocessing:

```python
 Load dataset
data = pd.read_csv('path/to/dataset.csv')

 Check for missing values
print(data.isnull().sum())

 Convert categorical 'Stress_Level' to numerical values
data['Stress_Level'] = data['Stress_Level'].map({'Low': 0, 'Moderate': 1, 'High': 2})

 Standardize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data.drop(columns=['GPA']))
```

## 2. Exploratory Data Analysis (EDA)

EDA helps us understand the dataset and discover relationships between variables. We visualize the distribution of GPA and examine correlations:

```python
 Plot GPA distribution
sns.histplot(data['GPA'], kde=True)
plt.title("Distribution of GPA")

 Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()
```

## 3. Model Training and Evaluation

We train various regression models and evaluate them using cross-validation. Here’s an example with Linear Regression:

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

model = LinearRegression()
scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
print("Cross-validated R2 Score for Linear Regression:", scores.mean())
```

Model Evaluation Metrics:

- R2 Score: Proportion of variance in GPA explained by the model.
- Mean Absolute Error (MAE): Average absolute difference between predicted and actual GPA.
- Mean Squared Error (MSE): Average squared difference, penalizing larger errors.

## 4. Hyperparameter Tuning

For the best-performing model (e.g., Random Forest), we use hyperparameter tuning to enhance accuracy.

```python
from sklearn.model_selection import RandomizedSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
grid_search = RandomizedSearchCV(RandomForestRegressor(), param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)
print("Best Hyperparameters:", grid_search.best_params_)
```

## 5. Model Interpretation with SHAP

Using SHAP values, we can interpret the impact of each lifestyle factor on GPA predictions:

```python
import shap
explainer = shap.TreeExplainer(grid_search.best_estimator_)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names=X.columns)
```

---

## Results

The table below summarizes the performance of each model based on cross-validation scores:

| Model             | R2 Score | Mean Absolute Error | Mean Squared Error |
| ----------------- | -------- | ------------------- | ------------------ |
| Linear Regression | 0.XX     | X.XX                | X.XX               |
| Ridge Regression  | 0.XX     | X.XX                | X.XX               |
| Lasso Regression  | 0.XX     | X.XX                | X.XX               |
| Random Forest     | 0.XX     | X.XX                | X.XX               |
| Gradient Boosting | 0.XX     | X.XX                | X.XX               |
| XGBoost           | 0.XX     | X.XX                | X.XX               |

# Insights

- Key Predictors: Factors such as study hours, stress level, and sleep hours show significant influence on GPA.
- Model Interpretability: SHAP values reveal which lifestyle choices are most impactful, helping students focus on areas for improvement.

## Future Enhancements

Future versions of this project could include:

- Adding more lifestyle factors for improved prediction accuracy.
- Experimenting with more complex models or neural networks.
- Using more sophisticated hyperparameter tuning techniques like Bayesian Optimization.

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a branch for your feature (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m 'Add YourFeature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or suggestions, please feel free to reach out or open an issue.
