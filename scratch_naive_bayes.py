# -*- coding: utf-8 -*-
"""Scratch Naive_Bayes.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1VTcXzglvsH6HwvSGL0JcIiFGWvZqiRvd
"""

import pandas as pd
import numpy as np

class NaiveBayes:
    def __init__(self):
        pass

    def fit(self, predictor, target):
        """
        Train the Naive Bayes model with training data.
        """
        # Convert target to pandas Series
        target_series = pd.Series(target)

        # Store unique classes and initialize dictionaries to store probabilities
        self.classes = target_series.unique()
        self.class_probs = {}
        self.feature_probs = {}

        # Calculate class probabilities
        for cls in self.classes:
            self.class_probs[cls] = np.mean(target_series == cls)

        # Calculate mean and standard deviation of features for each class
        for cls in self.classes:
            cls_data = predictor[target_series == cls]
            self.feature_probs[cls] = {
                'mean': cls_data.mean(axis=0),
                'std': cls_data.std(axis=0)
            }

    def _calculate_likelihood(self, x, mean, std):
        """
        Calculate the likelihood using Gaussian distribution.

        Parameters:
        -----------
        x : float
            Value to calculate the likelihood for.
        mean : float
            Mean value of the feature.
        std : float
            Standard deviation of the feature.

        Returns:
        --------
        likelihood : float
            Likelihood of the value given the mean and standard deviation.
        """
        exponent = np.exp(-((x - mean) ** 2) / (2 * std ** 2))
        return (1 / (np.sqrt(2 * np.pi) * std)) * exponent

    def predict(self, instance):
        """
        Predict the class label for a given instance.

        Parameters:
        -----------
        instance : array-like
            Features data for a single instance.

        Returns:
        --------
        prediction : object
            Predicted class label for the instance.
        """
        # Calculate posteriors
        posteriors = {}
        for cls in self.classes:
            prior = self.class_probs[cls]
            likelihood = 1
            for i, val in enumerate(instance):
                mean = self.feature_probs[cls]['mean'][i]
                std = self.feature_probs[cls]['std'][i]
                likelihood *= self._calculate_likelihood(val, mean, std)
            posteriors[cls] = prior * likelihood

        # Normalize posteriors
        total_posterior = sum(posteriors.values())
        normalized_posteriors = {cls: posterior / total_posterior for cls, posterior in posteriors.items()}

        # Make prediction
        prediction = max(normalized_posteriors, key=normalized_posteriors.get)
        return prediction

    def metrics_classification(self, predicted_labels, true_labels):
        """
        Calculate the accuracy, precision, and recall of the Naive Bayes model.

        Parameters:
        -----------
        predicted_labels : array-like
            Labels predicted by the model.
        true_labels : array-like
            True target labels.

        Returns:
        --------
        accuracy : float
            Model accuracy.
        precision : float
            Model precision.
        recall : float
            Model recall.
        """
        # Calculate the number of correct predictions, true positives, and false positives
        correct_predictions = 0
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for predicted_label, true_label in zip(predicted_labels, true_labels):
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
        accuracy = correct_predictions / len(true_labels)

        # Precision formula: TP / (TP + FP)
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0

        # Recall formula: TP / (TP + FN)
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

        return accuracy, precision, recall