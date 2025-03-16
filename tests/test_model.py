from src.train import train_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def test_model():
    # Simulate model training and evaluation
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Load data
    # X_train, X_test, y_train, y_test = ...
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    assert mse < 0.1, f"Test failed: Model's MSE is {mse}, which is higher than the threshold."
    print(f"Test passed: Model's MSE is {mse}.")
    
# Call test
test_model()
