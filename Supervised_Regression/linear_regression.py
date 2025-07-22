# Linear Regression Example: Predicting House Prices
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Generate synthetic data
np.random.seed(42)
sqft = np.random.randint(500, 4000, 100).reshape(-1, 1)
price = sqft * 150 + np.random.normal(0, 50000, size=sqft.shape)

# Split data
x_train, x_test, y_train, y_test = train_test_split(sqft, price, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(x_train, y_train)

# Predict
y_pred = model.predict(x_test)

# Evaluate
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Plot
plt.scatter(x_test, y_test, color='blue', label='Actual')
plt.plot(x_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel("Square Footage")
plt.ylabel("House Price")
plt.title("Linear Regression: House Price Prediction")
plt.legend()
plt.show()
