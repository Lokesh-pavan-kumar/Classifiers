# # Importing the required packages / libraries
#
# * Numpy is used to perform calculations on the datasets to
#   prepare the classifier.


import numpy as np  # For performing numerical operations on the data.
from typing import Union, List  # Used for type annotations


class BayesClassifier:
    """
    Bayes Classifier implementation :

    1. Identify the number of classes(c) in the data.
    2. For each class get the co-variance matrix(sigma_i) and the mean vector(mu_i).
    3. Prepare c discriminator functions.
    4. For an input x the assign argmax(i) of the discriminant functions.
    5. Return the predicted class-label.

    NOTE : For one-dimensional vectors such as mean and x(sample input) we are working with column vectors
    """

    def __init__(self):
        """Declares all the required instance variables used by the classifier"""
        self.X, self.y = None, None  # The set fit the classifier on, labels of the set
        self.m, self.n = None, None  # Number of examples(m), Number of features(n)
        self.num_classes = None  # Number of classes in the classification
        self.covariances = None  # Class-wise covariance matrices
        self.means = None  # Class-wise mean of the data
        self.priors = None  # Priors of the data
        print('Bayes Classifier initialized\n')

    def fit(self, X: np.ndarray, y: np.ndarray, equal_priors=False):
        """
        Initialises all the class variables with the required data

        :param X: The input data for classification
        :param y: The class labels for the input data
        :param equal_priors: If set to True the priors are taken equal irrespective of the underlying class frequencies
        """
        m, n = X.shape
        self.X = X
        self.y = y
        self.m, self.n = m, n
        self.num_classes = len(np.unique(y))
        self.covariances = self.get_covariance()  # Calculates the covariances of each class
        self.means = self.get_mean()  # Calculates the mean of each class

        if equal_priors:
            self.priors = [1 / self.num_classes] * self.num_classes
        else:
            # If equal_priors are not set then we calculate the priors from the data
            # numpy.unique when set return_counts as True returns the count of each unique
            # value in the array which can be used to find out the priors
            _, counts = np.unique(y, return_counts=True)
            counts = counts / np.sum(counts)  # If counts = [5, 4, 1] priors = [5/10, 4/10, 1/10]
            self.priors = counts.tolist()  # Converting the priors to a list

    def get_covariance(self):
        """
        Used to compute the covariance matrices
        :return: Covariance matrices of data grouped by classes
        """
        covariances = []
        for index in range(self.num_classes):
            indexes = np.where(self.y == index)
            class_matrix = self.X[indexes]
            cov = np.cov(class_matrix.T, ddof=0)
            assert cov.shape == (self.n,) * 2
            assert np.allclose(cov, cov.T)  # Covariance matrix is symmetric

            covariances.append(cov)
        return covariances

    def get_mean(self):
        """
        Used to compute the mean
        :return: Mean of data grouped by classes
        """
        means = []
        for index in range(self.num_classes):
            indexes = np.where(self.y == index)
            class_matrix = self.X[indexes]
            mean = np.mean(class_matrix, axis=0)
            assert self.n in mean.shape
            means.append(mean.reshape((-1, 1)))

        return means

    def get_disc(self, x: np.ndarray):
        """
        :param x: An input feature vector used for classification/prediction of label
        :return: Discriminant functions, g_i(x) for every class
        """
        discriminants = []  # This will hold the discriminant function for each class
        x = np.reshape(x, (-1, 1))  # Converting the input to a column vector
        for index in range(self.num_classes):
            disc = np.dot(
                np.dot((x - self.means[index]).T, np.linalg.inv(self.covariances[index])),
                (x - self.means[index])
            )
            if disc.ndim == 1:
                assert disc.shape == (1,)
            else:
                assert disc.shape == (1, 1)
            disc = -1 / 2 * disc.item()  # Converting disc into a scalar
            disc = disc - self.n / 2 * np.log(2 * np.pi) - 1 / 2 * np.log(
                np.linalg.det(self.covariances[index])) + np.log(self.priors[index])
            discriminants.append(disc)
        return discriminants

    def predict_class(self, x: np.ndarray, show_results=False):
        """
        :param x: The input pattern that needs to be classified/predicted.
        :param show_results: If set, then displays the discriminant values for the classification/prediction.
        :return: Integer value denoting the class the pattern belongs to.
        """
        scores = self.get_disc(x)
        if show_results:
            print('Scores for each class are: ', scores, sep='\n')
        return np.argmax(scores)


def get_accuracy_score(true_labels: Union[np.ndarray, List], predictions: Union[np.ndarray, List]):
    """
    Given two ground truth labels and the predictions returns the accuracy of the predictions
    with respect to those of the truth labels

    :param true_labels: The ground truth labels of the patterns
    :param predictions: The labels predicted by the classifier
    :return: Float denoting the accuracy of the classifier
    """
    num_patterns = len(predictions)
    return np.sum(true_labels == predictions) / num_patterns