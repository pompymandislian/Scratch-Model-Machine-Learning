# -*- coding: utf-8 -*-
"""Scratch k-nearest neighbor.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1NJLenbTgOFw36rpoxm-hcYwBNPwX3XEp
"""
import numpy as np
from collections import defaultdict

class Distance:

    def __init__(self, new_data, predictor):
        self.new_data = new_data
        self.predictor = predictor

    def manhattan_distance(self):
        """
        Function for calculate manhattan_distance

        Parameters:
        -----------
        new_data : int
            data new input

        predictor : list
            reference data for distance

        Returns:
        --------
        distance : int
            result of calculate manhattan_distance
        """
        # inital distance
        distance = 0

        # looping in data predictor
        for index in range(len(self.predictor)):

            # calculate manhattan distance
            manhattan_distance = np.sum(np.abs(self.new_data - self.predictor[index]))

            # sum to initial distance
            distance += manhattan_distance

        return distance

    def euclidean_distance(self):
        """
        Function for calculate euclidean_distance

        Parameters:
        -----------
        new_data : int
            data new input

        predictor : list
            reference data for distance

        Returns:
        --------
        distance : int
            result of calculate euclidean_distance
        """
        # inital distance
        distance = 0

        # looping in data predictor
        for index in range(len(self.predictor)):

            # calculate euclidean distance
            euclidean_distance = np.sqrt(np.sum((self.new_data - self.predictor[index]) ** 2))

            # sum to initial distance
            distance += euclidean_distance

        return distance

    def minkowski_distance(self, p):
      """
      Function for check minkowski_distance

      Parameters:
      -----------
      new_data : int
          data new input

      predictor : list
          reference data for distance

      p : int
        - p is 1 : manhattan_distance
        - p is 2 : euclidean_distance

      Return:
      -------
      distance : int
        result after calculate distance minkowski
      """
      # initial for manhattan and euclidean distance
      p_manhattan = 0
      p_euclidean = 0

      # condition if p is 1 or manhattan
      if p == 1:

        # looping in data predictor
        for index in range(len(self.predictor)):

            # calculate manhattan distance
            manhattan_distance = np.sum(np.abs(self.new_data - self.predictor[index]))

            # sum to initial distance
            p_manhattan += manhattan_distance

      # condition if p is 2 or euclidean
      elif p == 2:
        # looping in data predictor
        for index in range(len(self.predictor)):

            # calculate euclidean distance
            euclidean_distance = np.sqrt(np.sum((self.new_data - self.predictor[index]) ** 2))

            # sum to initial distance
            p_euclidean += euclidean_distance

      # inital for minkowski distance
      distance = 0

      # looping in data predictor
      for index in range(len(self.predictor)):

          # calculate mikowski distance
          minkowski_distance = (np.sum(np.abs(self.new_data - self.predictor[index])) ** p) ** 1 / p

          # sum to initial distance
          distance += minkowski_distance

      return distance

class KnnClassifier:
    """
    Implementation of classification using the K-Nearest Neighbors (KNN) method.
    """
    def __init__(self, distance_type='euclidean',
                 k_num=5, p=None, weight_method= 'uniform'):

        self.distance_type = distance_type
        self.k_num = k_num
        self.p = p
        self.weight_method = weight_method

    def fit(self, predictor, target):
        """
        Method for fitting data between predictor and target
        """
        self.predictor = predictor
        self.target = target

    def predict(self, new_data):
        """
        Perform prediction for new data using the KNN model.

        Parameters:
        -----------
        new_data : array-like
          New data to be predicted.

        Returns:
        --------
        predicted_labels : array-like
          Predicted labels for the new data.
        """
        distances = []
        for point in self.predictor:
            distance = Distance(new_data=new_data, predictor=point)

            if self.distance_type == 'manhattan':
                distances.append(distance.manhattan_distance())

            elif self.distance_type == 'euclidean':
                distances.append(distance.euclidean_distance())

            elif self.distance_type == 'minkowski':
                if self.p is None:
                    raise ValueError("Value of p must be provided for Minkowski distance calculation.")
                distances.append(distance.minkowski_distance(self.p))

            else:
                raise ValueError("Invalid distance type. Please choose from 'manhattan', 'euclidean', or 'minkowski'.")

        # Get indices of the k nearest neighbors
        nearest_neighbors = np.argsort(distances)[:self.k_num]

        # Get distances of the k nearest neighbors
        k_distances = [distances[i] for i in nearest_neighbors]

        # Calculate weights
        if self.weight_method == 'uniform':
            weights = np.ones_like(k_distances)

        elif self.weight_method == 'distance':
            weights = 1 / np.array(k_distances)

        else:
            raise ValueError("Invalid weight method. Please choose from 'uniform' or 'distance'.")

        # Get labels of the nearest neighbors
        neighbor_labels = [self.target[i] for i in nearest_neighbors]

        # Calculate weighted votes for each class
        weighted_votes = defaultdict(float)
        for label, weight in zip(neighbor_labels, weights):
            weighted_votes[label] += weight

        # Determine the majority label
        majority_label = max(weighted_votes.items(), key=lambda x: x[1])[0]

        return majority_label

    def metrics_classification(self, predicted_labels):
        """
        Calculate the accuracy, precision, and recall of the KNN model.

        Parameters:
        -----------
        predicted_labels : array-like
          Labels predicted by the model.

        Returns:
        --------
        accuracy : float
          Model accuracy.

        precision : float
          Model precision.

        recall : float
          Model recall.

        f1_score : float
            Model f1_score
        """
        # Calculate the number of correct predictions, true positives, and false positives
        correct_predictions = 0
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for predicted_label, true_label in zip(predicted_labels, self.target):
            if predicted_label == true_label:
                correct_predictions += 1
                if predicted_label == 1:
                    true_positives += 1
            else:
                if predicted_label == 1:
                    false_positives += 1
                if true_label == 1:
                    false_negatives += 1

        # Accuracy formula: (number of correct predictions) / (total predictions)
        accuracy = correct_predictions / len(self.target)

        # Precision formula: TP / (TP + FP)
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0

        # Recall formula: TP / (TP + FN)
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

        # F1 score formula: 2 * (precision * recall) / (precision + recall)
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return accuracy, precision, recall, f1_score

class KnnRegressor(KnnClassifier):
    def __init__(self, distance_type='euclidean',
                 k_num=5, p=None, weight_method='uniform'):
        super().__init__(distance_type, k_num, p, weight_method)

    def fit(self, predictor, target):
        """
        Train the KNN model with training data
        """
        self.predictor = predictor
        self.target = target

    def predict_regression(self, new_data):
        """
        Perform regression prediction for new data using the KNN model.

        Parameters:
        -----------
        new_data : array-like
          New data to be predicted.

        Returns:
        --------
        predicted_values : array-like
          Predicted values for the new data.
        """
        distances = []
        for point in self.predictor:
            distance = Distance(new_data=new_data, predictor=point)

            if self.distance_type == 'manhattan':
                distances.append(distance.manhattan_distance())

            elif self.distance_type == 'euclidean':
                distances.append(distance.euclidean_distance())

            elif self.distance_type == 'minkowski':
                if self.p is None:
                    raise ValueError("Value of p must be provided for Minkowski distance calculation.")
                distances.append(distance.minkowski_distance(self.p))

            else:
                raise ValueError("Invalid distance type. Please choose from 'manhattan', 'euclidean', or 'minkowski'.")

        # Get indices of the k nearest neighbors
        nearest_neighbors = np.argsort(distances)[:self.k_num]

        # Get distances of the k nearest neighbors
        k_distances = [distances[i] for i in nearest_neighbors]

        # Calculate weights
        if self.weight_method == 'uniform':
            weights = np.ones_like(k_distances)

        elif self.weight_method == 'distance':
            weights = 1 / np.array(k_distances)

        else:
            raise ValueError("Invalid weight method. Please choose from 'uniform' or 'distance'.")

        # Get values of the nearest neighbors
        neighbor_values = [self.target[i] for i in nearest_neighbors]

        # Calculate weighted average of neighbor values
        weighted_sum = sum(w * v for w, v in zip(weights, neighbor_values))
        weighted_average = weighted_sum / sum(weights)

        return weighted_average

    def metrics_regression(self, predicted_labels):
        """
        Calculate the mean_squared_error, root_mean_squared_error,
        mean_absolute_error, and r2 score of the KNN regression model.

        Parameters:
        -----------
        predicted_labels : array-like
            Predicted values by the model.

        Returns:
        --------
        mean_squared_error : float
            Mean squared error between true target values and predicted values.

        root_mean_squared_error : float
            Root mean squared error between true target values and predicted values.

        mean_absolute_error : float
            Mean absolute error between true target values and predicted values.

        r2 : float
            R-squared score (coefficient of determination) of the model.
            R-squared score provides a measure of how well the predictions approximate the true values.
            It ranges from 0 to 1, with 1 indicating perfect predictions.
        """
        # Calculate mean_squared_error
        mean_squared_error = np.mean((target - predicted_labels) ** 2)

        # Calculate root_mean_squared_error
        root_mean_squared_error = np.sqrt(mean_squared_error)

        # Calculate mean_absolute_error
        mean_absolute_error = np.mean(np.abs(target - predicted_labels))

        # Calculate R2
        total_variation = np.sum((target - np.mean(target)) ** 2)
        residual_variation = np.sum((target - predicted_labels) ** 2)
        r2 = 1 - (residual_variation / total_variation)

        return mean_squared_error, root_mean_squared_error, mean_absolute_error, r2
