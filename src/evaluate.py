from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(model, X_test, y_test):
    """ Evaluate the model using MSE and R2 score """
    y_pred = model.predict(X_test)
    
    # Calculate MSE and R²
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Output the evaluation metrics
    print(f"Mean Squared Error: {mse}")
    print(f"R-Squared: {r2}")
    
    return mse, r2
