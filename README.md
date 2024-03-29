# How to use this model:
---

## Supervised Model
---

### KNN Model Example

#### Classification:

  ```python
  # Inisialisasi model KNN
  knn_model = KnnClassifier()
  
  # Fitting model
  knn_model.fit(predictors, target)
  
  # Prediction label for each instance
  predicted_labels = np.array([knn_model.predict(data_point) for data_point in predictors])
  
  # Calculate accuracy, precision, recall, f1_score
  accuracy, precision, recall, f1_score = knn_model.metrics_classification(predicted_labels)
  
  print("Accuracy model KNN:", accuracy)
  print("Precision model KNN:", precision)
  print("Recall model KNN:", recall)
  print("F1_score model KNN:", f1_score)
```
#### Regressor:
```python
  # Inisialisasi model KNN
  knn_reg = KnnRegressor()
  
  # Fit model
  knn_reg.fit(predictors_reg, target_reg)
  
  # Prediction label for each instance
  predicted_labels_reg = np.array([knn_reg.predict(data_point) for data_point in predictors_reg])
  
  # Calculate mean_squared_error, root_mean_squared_error, mean_absolute_error, r2
  mean_squared_error, root_mean_squared_error, mean_absolute_error, r2 = knn_reg.metrics_regression(predicted_labels_reg)
  
  print("Mean Squared Error model KNN:", mean_squared_error)
  print("Root Mean Squared Error model KNN:", root_mean_squared_error)
  print("Mean Absolute Error model KNN:", mean_absolute_error)
  print("R2 Score model KNN:", r2)
```

### Naive Bayes Model Example

#### Classification:
```python
# Inisialisasi model Naive Bayes
naive_bayes = NaiveBayes()

# Fitting model
naive_bayes.fit(predictor, target)

# Prediction label for each instance
predicted_labels = np.array([naive_bayes.predict(data_point) for data_point in predictor.values])

# Calculate accuracy, precision, recall, f1_score
accuracy, precision, recall, f1_score = naive_bayes.metrics_classification(predicted_labels, target)

print("Accuracy model Naive Bayes:", accuracy)
print("Precision model Naive Bayes:", precision)
print("Recall model Naive Bayes:", recall)
print("F1_score model Naive Bayes:", f1_score)
```

### Linear Regression Model Example

#### Regressor:

```python
# Inisialisasi model LinearRegressor
regressor = LinearRegressor()

# Fitting model
regressor.fit(predictors, target)

# coef, intercept, and RSS
coef = regressor.coefficient() # obtain coef
intercept = regressor.intercept() # obtain intercept 
rss = regressor.residual_sum_square() # obtain rss

print('Intercept:', intercept)
print('Coefficient :', coef)
print('Residual Sum of Square :', RSS)

# Prediction label for each instance
predicted_labels_liner = regressor.predict(predictors)

# Calculate mean_squared_error, root_mean_squared_error, mean_absolute_error, r2
mean_squared_error, root_mean_squared_error, mean_absolute_error, r2 = regressor.metrics_regression(predicted_labels_liner)

print("Mean Squared Error model Linear Regression:", mean_squared_error)
print("Root Mean Squared Error model Linear Regression:", root_mean_squared_error)
print("Mean Absolute Error model Linear Regression:", mean_absolute_error)
print("R2 Score model Linear Regression:", r2)
```

### Logistic Regression Model Example

#### Classification:
```python
# Inisialisasi model LogisticRegression
logit_model = LogisticRegression(max_iter = 200, penalty = 'l1')

# Fitting model
logit_model.fit(predictors, target)

# Obtain cost function for look optimum model
logit_model.cost_function()

# Prediction label for each instance
prob_score = logit_model.predict(predictors)

# Calculate accuracy, precision, recall, f1_score
accuracy, precision, recall, f1_score = knn_model.metrics_classification(predicted_labels)

print("Accuracy model Logistic Regression:", accuracy)
print("Precision model Logistic Regression:", precision)
print("Recall model Logistic Regression:", recall)
print("F1_score model Logistic Regression:", f1_score)
```

### Decision Tree Model Example

#### Classification:
```python
# Inisialisasi model Decision Tree
model = ClassifierDesicionTree(max_depth = 3, max_features = 'auto')

# Fitting model
model.fit(predictors, target)

# Prediction label for each instance
predictions = model.predict(predictors)

# Calculate accuracy, precision, recall, f1_score
accuracy, precision, recall, f1_score = knn_model.metrics_classification(predicted_labels)

print("Accuracy model Decision Tree:", accuracy)
print("Precision model Decision Tree:", precision)
print("Recall model Decision Tree:", recall)
print("F1_score model Decision Tree:", f1_score)
```
