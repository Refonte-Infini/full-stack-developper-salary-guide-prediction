from sklearn.neural_network import MLPRegressor
import pymc3 as pm
import numpy as np
from transformers import pipeline

# 1. Inflation Adjustment Model
def inflation_adjusted_salary(nominal_salary, inflation_rate):
    return nominal_salary * (1 + inflation_rate / 100)

# 2. Compound Annual Growth Rate (CAGR)
def cagr(fv, pv, n):
    return ((fv / pv) ** (1 / n)) - 1

# 3. Neural Network Regression
def neural_network_regression():
    X = np.array([[1], [2], [3], [4]])
    y = np.array([70, 85, 100, 130])

    model = MLPRegressor(max_iter=500)
    model.fit(X, y)
    return model.predict([[5]])[0]

# 4. Bayesian Linear Regression
def bayesian_linear_regression():
    X = np.array([1, 2, 3, 4])
    y = np.array([70, 85, 100, 130])

    with pm.Model() as model:
        beta = pm.Normal("beta", mu=0, sigma=10)
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=1)
        y_pred = alpha + beta * X
        likelihood = pm.Normal("y", mu=y_pred, sigma=sigma, observed=y)
        trace = pm.sample(1000, return_inferencedata=False)
    return pm.summary(trace)

# 5. GPT (Text-Transforming)
def gpt_example():
    generator = pipeline("text-generation", model="gpt-2")
    output = generator("Predicting Full-Stack Developer salaries in 2025 involves", max_length=50)
    return output

# Running the Models
print("Inflation Adjusted Salary:", inflation_adjusted_salary(90000, 2.5))
print("CAGR:", cagr(120000, 90000, 3))
print("Neural Network Regression Prediction for 5 Years:", neural_network_regression())
print("Bayesian Linear Regression Trace Summary:")
print(bayesian_linear_regression())
print("GPT Example Output:")
print(gpt_example())
