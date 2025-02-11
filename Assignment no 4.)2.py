import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('weight-height.csv')

# Display first few rows of the data
print(df.head())

# Scatter plot of Height vs Weight
plt.scatter(df['Height'], df['Weight'], alpha=0.5)
plt.title('Height vs Weight')
plt.xlabel('Height (inches)')
plt.ylabel('Weight (pounds)')
plt.show()

# Prepare data for MLP Regression
X = df[['Height']]  # Independent variable (Height)
y = df['Weight']    # Dependent variable (Weight)

# Step 1: Create and fit the Multi-layer Perceptron model
mlp_model = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=1000, random_state=42)
mlp_model.fit(X, y)

# Step 2: Make predictions using the MLP model
y_pred = mlp_model.predict(X)

# Step 3: Plot the results
plt.scatter(df['Height'], df['Weight'], alpha=0.5, label='Actual Data')
plt.scatter(df['Height'], y_pred, color='red', alpha=0.5, label='Predictions')
plt.title('Height vs Weight with MLP Predictions')
plt.xlabel('Height (inches)')
plt.ylabel('Weight (pounds)')
plt.legend()
plt.show()

