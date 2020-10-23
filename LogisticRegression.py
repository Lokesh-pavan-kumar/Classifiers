# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.style.use('fivethirtyeight')

# In[2]:


train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# In[3]:


print(f'Shape of the training data : {train_df.shape}')
print(f'Shape of the test data : {test_df.shape}')

# In[4]:


iris_columns = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width', 'Class']
train_df.columns = iris_columns
test_df.columns = iris_columns

class_values = sorted(train_df['Class'].unique().tolist())
class_to_idx = {class_name: idx for idx, class_name in enumerate(class_values)}
idx_to_class = {idx: class_name for idx, class_name in enumerate(class_values)}

# In[5]:


train_df.replace(to_replace={'Class': class_to_idx}, inplace=True)
test_df.replace(to_replace={'Class': class_to_idx}, inplace=True)

# In[6]:


X_train = train_df.iloc[:, :-1].values
y_train = train_df['Class'].values

X_test = test_df.iloc[:, :-1].values
y_test = test_df['Class'].values

# In[7]:


print(f'Training set, Feature vectors : {X_train.shape} and Labels : {y_train.shape}')
print(f'Test set, Feature vectors : {X_test.shape} and Labels : {y_test.shape}')


# In[8]:


def sigmoid(X: np.ndarray):
    return 1 / (1 + np.exp(-X))


def normalize(X: np.ndarray):
    return (X - X.mean(axis=0)) / X.std(axis=0)


class LogisticRegression:
    """
    Logistic Regression Numpy implementation

    1. Identify the size of the data i.e number of features(m) and dimensionality (n).
    2. Initialize a weight matrix of size (n, 1) and a scalar quantity bias set to 0.0
    3. Perform classification using binary_crossentropy loss function and iterate until and optimum
       loss is reached.
    4. Minimization algorithm is gradient descent, number of epochs may be fixed.

    NOTE : For one-dimensional vectors such as mean and x(sample input) we are working with column vectors.
    """

    def __init__(self):
        """Declares all the required instance variables used by the classifier"""
        self.X = None  # The feature vectors [shape = (m, n) => (n, m)]
        self.y = None  # The regression outputs [shape = (m, 1)]
        self.W = None  # The parameter vector `W` [shape = (n, 1)]
        self.bias = None
        self.lr = None  # Learning Rate `alpha`
        self.m = None
        self.n = None
        self.epochs = None
        print('Logistic Regression initialized')

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 50,
            use_bias: bool = True, lr: float = 0.02, verbose: bool = True):
        """
        Initialises all the class variables with the required data

        :param X: The input data for classification
        :param y: The class labels for the input data
        :param epochs: The number of epochs for gradient_descent
        :param use_bias: Boolean to specify whether or not to use bias
        :param lr: The learning rate
        :param verbose: A boolean if set to True, indicates the progress of the training
        """
        (self.m, self.n) = X.shape

        # Normalising the data
        self.X = normalize(X)
        assert y.shape == (self.m, 1) or (self.m,)
        if y.ndim == 1:
            self.y = np.expand_dims(y, axis=1)
        self.W = np.random.random((self.n, 1)) * 1e-3
        if use_bias:
            self.bias = 0.0
        self.epochs = epochs
        self.lr = lr
        return self.minimize(verbose=verbose)

    def minimize(self, verbose: bool, epsilon=1e-6):
        """
        :param verbose: If set to true returns the progress of the training after every epoch
        :param epsilon: A very small floating point used to smoothing
        :return: A dictionary containing the history for loss, accuracy, bias values and
                 a numpy array containing history of weights
        """
        cost_hist = []
        accuracy_hist = []
        bias_hist = []
        weight_hist = []

        for num_epoch in range(self.epochs):
            if self.bias is None:  # If use_bias is false
                predictions = sigmoid(np.dot(self.X, self.W))  # h(x) = 1/1+exp(- dot_product(X * W))
            else:  # If use_bias is true
                predictions = sigmoid(np.dot(self.X, self.W) + self.bias)  # h(x) = 1/1+exp(- dot_product(X * W + bias))

            # When the predictions are exactly equal to 1 or 0 using np.log on them gives us a runtime warning
            # To prevent that behaviour we add/subtract a very small number(epsilon) from the predictions
            predictions[predictions == 0] = predictions[predictions == 0] + 1e-7
            predictions[predictions == 1] = predictions[predictions == 1] - 1e-7

            assert predictions.shape == self.y.shape
            grad_w = np.dot(self.X.T, (predictions - self.y))  # Gradient of X wrt to the cost function
            self.W = self.W - self.lr * grad_w  # Updating the weight matrix
            if self.bias is not None:  # If bias is used, we update the bias too
                grad_b = np.sum(predictions - self.y)
                self.bias = self.bias - self.lr * grad_b  # Updating the bias
            assert self.W.shape == grad_w.shape

            # Binary Cross Entropy cost
            cost = (-1 / self.m) * (self.y * np.log(predictions) + (1 - self.y) * np.log(1 - predictions))
            cost = np.sum(cost)
            # Calculating the accuracy
            acc = self.get_accuracy(predictions)

            cost_hist.append(cost)
            accuracy_hist.append(acc)
            bias_hist.append(self.bias)
            weight_hist.append(self.W)

            if verbose:  # If verbose is set to true, we show the progress of the training
                print(f'Epoch : {num_epoch + 1}/{self.epochs}\tLoss : {cost}\tAccuracy : {acc}')

        return dict(loss=cost_hist, accuracy=accuracy_hist, bias=bias_hist), weight_hist

    def get_accuracy(self, predictions):
        predictions = predictions > 0.5
        return np.mean(predictions == self.y)

    def predict(self, x: np.ndarray):
        x = (x - x.mean(axis=0)) / x.std(axis=0)
        if self.bias is None:
            return sigmoid(np.dot(x, self.W))
        else:
            return sigmoid(np.dot(x, self.W) + self.bias)

    def evaluate(self, X: np.ndarray, y: np.ndarray):
        if y.ndim == 1:
            y = np.expand_dims(y, axis=1)
        predictions = self.predict(X)
        predictions = predictions > 0.5
        return np.mean(predictions == y)


# In[9]:


logistic_regression = LogisticRegression()
history, weights_hist = logistic_regression.fit(X_train, y_train, 50, lr=0.01, verbose=False)
weights_hist = np.array(weights_hist)

# In[10]:


df = pd.DataFrame(history)

# In[11]:


df.head()

# In[12]:


df.plot(y=['bias', 'accuracy', 'loss'])

# # Data Exploration

# In[13]:


train_df.head()

# In[14]:


class_1 = train_df['Class'] == 0
class_2 = train_df['Class'] == 1

fig, axes = plt.subplots(nrows=1, ncols=2)

train_df[class_1].plot.scatter(x='Sepal_Length', y='Sepal_Width',
                               color='red', label='Iris-Versicolor', figsize=(8, 6), ax=axes[0])

train_df[class_2].plot.scatter(x='Sepal_Length', y='Sepal_Width',
                               color='blue', label='Iris-Virginica', ax=axes[0])

train_df[class_1].plot.scatter(x='Petal_Length', y='Petal_Width',
                               color='red', label='Iris-Versicolor', figsize=(8, 6), ax=axes[1])

train_df[class_2].plot.scatter(x='Petal_Length', y='Petal_Width',
                               color='blue', label='Iris-Virginica', ax=axes[1])

axes[0].set_title('Classes Vs. Sepal-Dimensions')
axes[1].set_title('Classes Vs. Petal-Dimensions')

plt.show()

# In[15]:
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))

train_df['Sepal_Length'].plot.hist(edgecolor='black', linewidth=1.5,
                                   title='Sepal_Length Hist', ax=axes[0, 0])

train_df['Sepal_Width'].plot.hist(edgecolor='black', linewidth=1.5,
                                  title='Sepal_Width Hist', ax=axes[0, 1])

train_df['Petal_Length'].plot.hist(edgecolor='black', linewidth=1.5,
                                   title='Petal_Length Hist', ax=axes[1, 0])

train_df['Petal_Width'].plot.hist(edgecolor='black', linewidth=1.5,
                                  title='Petal_Width Hist', ax=axes[1, 1])

plt.tight_layout()
plt.show()
# In[16]:
ax = train_df['Class'].value_counts().plot.bar(
    title='Class Counts',
    color=['#fc4f30', '#008fd5'],
    figsize=(8, 6)
)

ax.set_xlabel('Classes')
ax.set_ylabel('Frequencies')

# In[34]:

fig = plt.figure()
ax = plt.gca()
x = list(range(50))
y = weights_hist[:, 3]


def update_point(n, x, y):
    plt.cla()
    # The plot the the weight
    plt.plot(x, y, color='blue', linewidth=1.2)
    plt.plot(x[n], y[n], 'o', color='red')
    # Put the value of the weight
    plt.text(x[n], y[n], f'{y[n].item():.3f}')


ani = FuncAnimation(fig, update_point, frames=500, fargs=(x, y))
plt.show()
