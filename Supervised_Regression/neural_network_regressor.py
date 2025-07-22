# Neural Network Regressor
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X, y = make_regression(n_samples=100, n_features=1, noise=15)
X_train, X_test, y_train, y_test = train_test_split(X, y)

model = MLPRegressor(hidden_layer_sizes=(10,), max_iter=1000)
model.fit(X_train, y_train)
preds = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, preds))
