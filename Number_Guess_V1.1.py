# Manual implementation - Does work

# from PIL import Image
import numpy as np
import math
from sklearn.model_selection import train_test_split
# Image loading will do this later - add a function
#im = Image.open("image.png")
#pic = im.load()
# pix[3,3]

# static setting of "train" "test" but will be revised
# target should only exist for train data
from matplotlib import pyplot as plt


class list_holder():
    temp_list = []
    
    def __init__(self,temp_list):
        self.temp_list = temp_list


#Network class
class NeuralNetwork():
    weight_bias = {}
    calc_layers = {}
    layers = []
    learning_rate = 0.0
    iterations = 0
    loss = []
    input = list()
    train_targets = list()

    def __init__(self, layers=[15, 30, 10], learning_rate=0.017, iterations=500):
        self.layers = layers
        self.learning_rate = learning_rate
        self.iterations = iterations

    # Random assignment of weight and bias
    def start_weights(self):
        np.random.seed(1)
        for i in range(len(self.layers)-1):
            self.weight_bias['w'+str(i+1)] = -1 * np.random.randn(self.layers[i], self.layers[i+1]) - .6
            self.weight_bias['b' + str(i+1)] = -1 * np.random.randn(self.layers[i+1]) - .6

    # Activation funciton - compares a value with 0
    # Returns the 0 or number * .01
    def leaky_relu(self, x):
        return np.where(x > 0, x, x * 0.01)

    # Leaky relu deravative
    def relu_derivative(self, x, alpha=0.01):
        dx = np.ones_like(x)
        dx[x < 0] = alpha
        return dx

    # Activation funciton does the function below
    # Value from 0 to 1
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Calculate the derivative of sigmoid output - per nueron output
    def sigmoid_derivative(self, output):
        return output * (1.0 - output)

    # Activation function
    def softmax(self ,x):
        expX = np.exp(x)
        return expX / expX.sum()

    #Softmax derivative
    def softmax_derivative(self, pred):
        pred_diagonal = np.diag(pred)
        for i in range(len(pred_diagonal)):
            for j in range(len(pred_diagonal)):
                if i == j:
                    pred_diagonal[i][j] = pred[i] * (1 - pred[i])
                else:
                    pred_diagonal[i][j] = -pred[i] * pred[j]
        return pred_diagonal

    # Cross entropy function for binary loss
    def entropy_loss(self, actual, pred):
        samples = len(actual)
        loss = -1 / samples * (np.sum(np.multiply(np.log(pred), actual) + np.multiply((1 - actual), np.log(1 - pred))))
        return loss

    # Multi class entropy
    def cross_entropy(self, y, p):
        m = y.shape[0]
        log_likelihood = []
        for i in range(m):
            for j in range(y.shape[1]):
                if y[i,j] == 1:
                    log_likelihood.append(-math.log(p[i, j], 10))
                else:
                    log_likelihood.append(-math.log(1-p[i, j], 10))
        loss = np.sum(log_likelihood)
        return loss / m

    def convert(self, prediction):
        max = np.amax(prediction)
        prediction2 = []
        for i in prediction[0]:
            if i == max:
                prediction2.append(1)
            else:
                prediction2.append(0)
        return prediction2

    # Forward propagation - moving from input to output in network
    # loop optimize?
    def forward_propagation(self):
        # Creation of the hidden layer
        first_calc = np.dot(self.input, self.weight_bias['w1']) + self.weight_bias['b1']
        # Activing each member of hidden layer
        first_activation = self.leaky_relu(first_calc)
        # Creation of output layer
        output_layer = first_activation.dot(self.weight_bias['w2']) + self.weight_bias['b2']
        # Activation of output
        prediction = self.sigmoid(output_layer)
        loss = self.entropy_loss(self.train_targets, prediction)

        # save the calc layers
        self.calc_layers['1I'] = first_calc
        self.calc_layers['1H'] = first_activation
        self.calc_layers['1O'] = output_layer

        return prediction, loss
    
    # Backward Propagation
    def backward_propagation(self, prediction):
        dloss_predict = -(np.divide(self.train_targets, prediction) - np.divide((1 - self.train_targets), (1 - prediction)))
        dloss_sigmoid = self.sigmoid_derivative(prediction)
        dloss1O = dloss_sigmoid*dloss_predict

        dloss1H = dloss1O.dot(self.weight_bias['w2'].T)
        dlossw2 = self.calc_layers['1H'].T.dot(dloss1O)
        dlossb2 = np.sum(dloss1O, axis=0)

        dloss1I = dloss1H * self.relu_derivative(self.calc_layers['1I'])
        dlossw1 = np.dot(self.input.T, dloss1I)
        dlossb1 = np.sum(dlossw1, axis=0)

        self.weight_bias['w1'] = self.weight_bias['w1'] - self.learning_rate * dlossw1
        self.weight_bias['w2'] = self.weight_bias['w2'] - self.learning_rate * dlossw2
        self.weight_bias['b1'] = self.weight_bias['b1'] - self.learning_rate * dlossb1
        self.weight_bias['b2'] = self.weight_bias['b2'] - self.learning_rate * dlossb2

    def fit(self, input, actual):
        self.start_weights()  # initialize weights and bias

        self.input = list()
        self.train_targets = list()

        self.input = np.array(input)
        self.train_targets = np.array(actual)

        for i in range(self.iterations):
            prediction, loss = self.forward_propagation()
            self.backward_propagation(prediction)
        #    print(loss," ",self.convert(prediction))
            self.loss.append(loss)
            
            
    def predict(self,input, actual=[]):
        first_calc = np.dot(input, self.weight_bias['w1']) + self.weight_bias['b1']
        first_activation = self.leaky_relu(first_calc)
        output_layer = first_activation.dot(self.weight_bias['w2']) + self.weight_bias['b2']
        prediction = self.sigmoid(output_layer)
        return np.argmax(prediction)
        
    def plot_loss(self):
        width = 12
        height = 10
        plt.figure(figsize=(width, height))
        plt.plot(np.linspace(0, self.iterations, len(self.loss)), np.array(self.loss),label='Loss ')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()


input_set = [[1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1],  # 0
             [1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],  # 0
             [1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0],  # 0

             [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],  # 1
             [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],  # 1
             [1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1],  # 1

             [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1],  # 2
             [0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1],  # 2
             [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0],  # 2

             [1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],  # 3
             [0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1],  # 3
             [1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0],  # 3

             [1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1],  # 4
             [1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],  # 4
             [1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1],  # 4

             [1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1],  # 5
             [1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1],  # 5
             [1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0],  # 5

             [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1],  # 6
             [1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1],  # 6
             [1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1],  # 6

             [1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],  # 7
             [1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],  # 7
             [1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],  # 7

             [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],  # 8

             [1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1],  # 9
             [1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0],  # 9
             [1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1]]  # 9


target = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],

          [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],

          [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],

          [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],

          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],

          [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],

          [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],

          [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],

          [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],

          [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]


def print_num(input):
    s = ""
    for i in range(1,16):
        s += " " + str(input[i - 1])
        if i%3 == 0:
            print(s)
            s = ""


# for x in range(len(input_set)):
#    print(np.argmax(target[x]))
#    print_num(input_set[x])
#    print('\n')

            
# Turning matrix above into np arrays
X = np.array(input_set)
Y = np.array(target)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0, test_size=.24)
NN = NeuralNetwork()
NN.fit(X_train, Y_train)
NN.plot_loss()

num_correct = 0
for x in range(len(X)):
    pred = NN.predict(X[x])
    print(pred," ",np.argmax(Y[x]))
    if pred == np.argmax(Y[x]):
        num_correct += 1

print('The acc was: ', (num_correct/len(Y)))

pred = NN.predict([0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1]) # 3
print(pred)