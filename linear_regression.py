import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample data: Hours studied vs Test Score
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1, 1)
y = np.array([35, 40, 50, 55, 65, 70, 75, 80, 90])

# Train the model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Print results
print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_[0])

# Plot
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, y_pred, color='red', label='Predicted')
plt.xlabel("Hours Studied")
plt.ylabel("Test Score")
plt.title("Linear Regression: Test Score vs Hours Studied")
plt.legend()
plt.grid(True)
plt.show()
