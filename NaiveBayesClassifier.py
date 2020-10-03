import numpy as np


class NaiveBayesClassifier:
    def __init__(self):
        """Declares all the required instance variables used by the classifier"""
        self.X, self.y = None, None  # The set fit the classifier on, labels of the set
        self.m, self.n = None, None  # Number of examples(m), Number of features(n)
        self.num_classes = None  # Number of classes in the classification
        self.standard_deviations = None  # Class-wise covariance matrices
        self.means = None  # Class-wise mean of the data
        self.priors = None  # Priors of the data
        print('Naive Bayes Classifier initialized\n')

    def fit(self, X: np.ndarray, y: np.ndarray, equal_priors: bool = True):
        m, n = X.shape
        self.X = X
        self.y = y
        self.m, self.n = m, n
        self.num_classes = len(np.unique(y))
        # Calculates the mean and variances of each class
        self.means, self.standard_deviations = self.get_mean_and_variance()

        if equal_priors:
            self.priors = [1 / self.num_classes] * self.num_classes
        else:
            # If equal_priors are not set then we calculate the priors from the data
            # numpy.unique when set return_counts as True returns the count of each unique
            # value in the array which can be used to find out the priors
            _, counts = np.unique(y, return_counts=True)
            counts = counts / np.sum(counts)  # If counts = [5, 4, 1] priors = [5/10, 4/10, 1/10]
            self.priors = counts.tolist()  # Converting the priors to a list

    def get_mean_and_variance(self):
        means = np.empty((self.num_classes, self.n))
        standard_deviations = np.empty((self.num_classes, self.n))
        for index in range(self.num_classes):
            class_examples = self.X[self.y == index]
            means[index] = np.mean(class_examples, axis=0)
            standard_deviations[index] = np.std(class_examples, axis=0)

        return means, standard_deviations

    def approximate_normal(self, x: np.ndarray):
        assert self.n in x.shape
        x = np.squeeze(x)
        exponent = -1 / 2 * np.square((x - self.means)) / np.square(self.standard_deviations)
        coefficient = 1 / (self.standard_deviations * np.sqrt(2 * np.pi))
        return coefficient * np.exp(exponent)

    def predict_class(self, x: np.ndarray):
        probabilities = self.approximate_normal(x)
        probabilities = np.multiply(probabilities, np.reshape(self.priors, (-1, 1)))
        probabilities = np.prod(probabilities, axis=1)
        return np.argmax(probabilities)