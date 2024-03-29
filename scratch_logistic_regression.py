# -*- coding: utf-8 -*-
"""scratch_logistic_regression.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1iCtYFbp1VXVKlTkjjwFYcb_nemIPyi2W
"""

class LogisticRegression:
    """
    Implementing logistic regression using gradient descent optimization.

    Parameters
    ----------
    max_iter : int, default=1000
        The maximum number of iterations to be performed during training.

    penalty : {'l1', 'l2', 'elasticnet', None}, default=None
        The type of regularization to be used. 'l1' for Lasso, 'l2' for Ridge,
        'elasticnet' for a combination of Lasso and Ridge, None for no regularization.

    lambda_val : float, default=0.1
        The regularization strength to be applied if regularization is used.

    Attributes
    ----------
    max_iter : int
        The maximum number of iterations to be performed during training.

    penalty : {'l1', 'l2', 'elasticnet', None}
        The type of regularization to be used.

    lambda_val : float
        The regularization strength to be applied if regularization is used.
    """

    def __init__(self, max_iter=1000, penalty=None, lambda_val=0.1):
        self.max_iter = max_iter
        self.penalty = penalty
        self.lambda_val = lambda_val

    def fit(self, predictor, target):
        """
        Train the logistic regression model.

        Parameters
        ----------
        predictor : array-like, shape (n_samples, n_features)
            Training feature data.

        target : array-like, shape (n_samples,)
            Training target data.
        """
        self.predictor = predictor
        self.target = target

    def likelihood(self):
        """
        Calculate the log-likelihood of the model.

        Returns
        -------
        log_likelihood : float
            The log-likelihood of the model.
        """
        # threshold
        p_success = np.mean(self.target)  # Mean of the target as the success probability

        # obtain data success and failed
        n_success = np.sum(self.target == 1)  # success
        n_failed = np.sum(self.target == 0)  # failed

        # calculate likelihood
        likelihood = (p_success**n_success) * ((1 - p_success)**n_failed)

        # Calculate log-likelihood
        log_likelihood = n_success * np.log(p_success) + n_failed * np.log(1 - p_success)

        return log_likelihood

    def initialize_parameter(self):
        """
        Initialize the model parameters.

        Returns
        -------
        b0 : float
            Initial intercept.

        b1 : array-like, shape (n_features,)
            Initial coefficients.
        """
        # Extract the number of predictors
        n_parameters = self.predictor.shape[1]

        # Initialize the parameter estimate
        b0_initial = 0.0
        b1_initial = np.zeros(n_parameters)

        return b0_initial, b1_initial

    def sigmoid(self):
        """
        Calculate the success probability using the sigmoid function.

        Returns
        -------
        pi : array-like, shape (n_samples,)
            The success probability.
        """
        # Initialize b0 and b1
        b0, b1 = self.initialize_parameter()

        # Calculate the logit value
        logit = b0 + np.dot(self.predictor, b1)

        # Calculate the success probability
        pi = np.exp(logit) / (1+np.exp(logit))

        return pi

    def cost_function(self, eps=1e-10):
        """
        Calculate the cost function (log loss) of the model.

        Parameters
        ----------
        eps : float, optional
            Additional value to avoid log of zero.

        Returns
        -------
        log_loss : float
            The log loss of the model.
        """
        # Calculate the success probability
        pi = self.sigmoid()

        # Calculate the log-likelihood value when y=1
        log_like_success = self.target * np.log(pi + eps)

        # Calculate the log-likelihood value when y=0
        log_like_failure = (1 - self.target) * np.log(1 - pi + eps)

        # Calculate the negative log-likelihood or log loss
        log_like_total = log_like_success + log_like_failure

        # log loss
        log_loss = -np.sum(log_like_total)

        return log_loss

    def gradient_b0_b1(self):
        """
        Calculate the gradient of the cost function with respect to the model parameters.

        Returns
        -------
        grad_b0 : float
            The gradient with respect to the intercept.

        grad_b1 : array-like, shape (n_features,)
            The gradient with respect to the coefficients.
        """
        # obtain function sigmoid
        pi = self.sigmoid()

        # Calculate the gradient of log loss w.r.t b0
        grad_b0 = np.sum(pi-self.target)

        # Calculate the gradient of log loss w.r.t b0
        grad_b1 = np.dot((self.predictor.T),(pi-self.target))

        return grad_b0, grad_b1

    def gradient_descent(self, eta=0.01, tol=1e-4):
        """
        Perform optimization using gradient descent.

        Parameters
        ----------
        eta : float, optional
            The learning rate.

        tol : float, optional
            The tolerance to stop iteration.

        Returns
        -------
        b0 : float
            The optimal estimate of the intercept.

        b1 : array-like, shape (n_features,)
            The optimal estimate of the coefficients.

        iterations : int
            The number of iterations performed.

        log_loss : array-like, shape (iterations,)
            The log loss value at each iteration.
        """
        # Initialize parameters
        b0, b1 = self.initialize_parameter()

        # Store log loss values
        log_loss = []

        # Perform gradient descent
        for i in range(self.max_iter):
            # Calculate sigmoid
            pi = self.sigmoid()

            # Calculate log loss
            loss = self.cost_function()
            log_loss.append(loss)

            # Calculate gradients
            grad_b0, grad_b1 = self.gradient_b0_b1()

            # calculate regulation
            regulation_l1 = self.lambda_val * np.sign(b1) # Lasso
            regulation_l2 = self.lambda_val * b1         # Ridge

            # Apply regularization if specified
            if self.penalty == 'l1':
                grad_b1 += regulation_l1

            elif self.penalty == 'l2':
                grad_b1 += regulation_l2

            elif self.penalty == 'elasticnet':
                grad_b1 += regulation_l1 + regulation_l2

            # Update parameters
            b0 -= eta * grad_b0
            b1 -= eta * grad_b1

            # Check convergence
            if np.all(np.abs(grad_b0) < tol) and np.all(np.abs(grad_b1) < tol):
                break

        return b0, b1, i+1, log_loss

    def predict(self, new_data):
        """
        Predict the target labels for new data.

        Parameters
        ----------
        new_data : array-like, shape (n_samples, n_features)
            New data to be predicted.

        Returns
        -------
        predictions : array-like, shape (n_samples,)
            Predicted target labels.
        """
        b0, b1, _, _ = self.gradient_descent()
        predictions = []

        for data_point in new_data:
            # Calculate the linear combination
            linear_combination = b0 + np.dot(data_point, b1)

            # Using the logistic function to calculate probabilities
            probability = 1 / (1 + np.exp(-linear_combination))

            # Threshold probabilities to get binary predictions
            prediction = 1 if probability > 0.5 else 0
            predictions.append(prediction)

        return np.array(predictions)

    def metrics_classification(self, predicted_labels):
        """
        Calculate the accuracy, precision, and recall of the logistic regression model.

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
        correct_predictions = np.sum(predicted_labels == self.target)
        true_positives = np.sum((predicted_labels == 1) & (self.target == 1))
        false_positives = np.sum((predicted_labels == 1) & (self.target == 0))
        false_negatives = np.sum((predicted_labels == 0) & (self.target == 1))

        # Accuracy formula: (number of correct predictions) / (total predictions)
        accuracy = correct_predictions / len(self.target)

        # Precision formula: TP / (TP + FP)
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0

        # Recall formula: TP / (TP + FN)
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

        # F1 score formula: 2 * (precision * recall) / (precision + recall)
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return accuracy, precision, recall, f1_score