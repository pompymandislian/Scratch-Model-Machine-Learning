How to used this model:

## KNN Model Example

### Classification:

  ```python
  # Inisialisasi model KNN
  knn_model = KnnClassifier()
  
  # Fitting model
  knn_model.fit(predictors, target)
  
  # Prediction label
  predicted_labels = np.array([knn_model.predict(data_point) for data_point in predictors])
  
  # Calculate accuracy, precision, recall
  accuracy, precision, recall = knn_model.metrics_classification(predicted_labels)
  
  print("Accuracy model KNN:", accuracy)
  print("Precision model KNN:", precision)
  print("Recall model KNN:", recall)
```
### Regressor:
```python
  # Create objek KnnRegressor
  knn_reg = KnnRegressor()
  
  # Fit model
  knn_reg.fit(predictors_reg, target_reg)
  
  # Prediction label
  predicted_labels_reg = np.array([knn_reg.predict(data_point) for data_point in predictors_reg])
  
  # Calculate mean_squared_error, root_mean_squared_error, mean_absolute_error, r2
  mean_squared_error, root_mean_squared_error, mean_absolute_error, r2 = knn_reg.metrics_regression(predicted_labels_reg)
  
  print("Mean Squared Error model KNN:", mean_squared_error)
  print("Root Mean Squared Error model KNN:", root_mean_squared_error)
  print("Mean Absolute Error model KNN:", mean_absolute_error)
  print("R2 Score model KNN:", r2)
```