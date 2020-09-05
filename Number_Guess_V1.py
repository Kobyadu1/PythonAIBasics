# First version - Does not work because I tried using softmax - only keeping for history

# from PIL import Image
import numpy as np
import math
# Image loading will do this later - add a function
#im = Image.open("image.png")
#pic = im.load()
# pix[3,3]

#static setting of "train" "test" but will be revised
#target should only exist for train data


#Network class
class NeuralNetwork():
    weight_bias = {}
    calc_layers = {}
    layers = []
    learning_rate = 0.0
    iterations = 0
    loss = []
    input = None
    train_targets = None

    def __init__(self, input, train_targets, layers=[15, 10, 10], learning_rate=0.001, iterations=100):
        self.layers = layers
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.input = input
        self.train_targets = train_targets

    # Random assignment of weight and bias
    def start_weights(self):
        np.random.seed(1)
        for i in range(len(self.layers)-1):
            self.weight_bias['w'+str(i+1)] = np.random.randn(self.layers[i], self.layers[i+1])
            self.weight_bias['b' + str(i+1)] = np.random.randn(self.layers[i+1])

    # Activation funciton - compares a value with 0
    # Returns the 0 or number * .01
    def leaky_relu(self, x):
        max = np.maximum(0, x)
        if max > 0:
            return max * .01
        else:
            return np.zeros(self.layers[0])

    # Leaky relu deravative
    def relu_derivative(self ,dl, x, alpha=0.01):
        dx = np.ones_like(x)
        dx[x < 0] = alpha
        return dx * dl

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

    # Loss function - binary loss idk this yet - segmoid
    def entropy_loss(self, actual, predict):
        loss = -1 / (np.sum(np.multiply(np.log(predict), actual) + np.multiply((1 - actual), np.log(1 - predict))))
        return loss

    # Idk if this is right
    def cross_entropy(self, p, y):
        m = y.shape[0]
        log_likelihood = []
        for i in range(m):
            if y[i] == 1:
                log_likelihood.append(-math.log(p[i], 10))
            else:
                log_likelihood.append(-math.log(1-p[i], 10))
        loss = np.sum(log_likelihood) / m
        return loss

    # Accuracy finder
    def acc(self, y, pred):
        acc = int(sum(y == pred) / len(y) * 100)
        return acc

    def convert(self,prediction):
        max = np.max(prediction)
        prediction2 = []
        for i in prediction:
            if i == max:
                prediction2.append(1)
            else:
                prediction2.append(0)
        return prediction2

    # Forward propagation - moving from input to output in network
    # loop optimize?
    def forward_propagation(self):
        # Creating of the hidden layer
        first_calc = np.dot(self.input,self.weight_bias['w1']) + self.weight_bias['b1']
        # Activing each member of hidden layer
        first_activation = self.sigmoid(first_calc)
        # Creation of output layer
        output_layer = first_activation.dot(self.weight_bias['w2']) + self.weight_bias['b2']
        # Activation of output
        prediction = self.softmax(output_layer)

        loss = self.cross_entropy(prediction, self.train_targets)

        # save the calc layers
        self.calc_layers['1I'] = first_calc
        self.calc_layers['1H'] = first_activation
        self.calc_layers['1O'] = output_layer

        return prediction, (loss/10)
    # Backward Propagation
    def backward_propagation(self, prediction):
        dloss_predict = -(np.divide(self.train_targets, prediction) - np.divide((1 - self.train_targets), (1 - prediction)))
        dloss_softmax = self.softmax_derivative(prediction)
        dloss1O = dloss_softmax*dloss_predict

        dloss1H = dloss1O.dot(self.weight_bias['w2'].T)
        dlossw2 = self.calc_layers['1H'].T.dot(dloss1O)
        dlossb2 = np.sum(dloss1O, axis=0)

        dloss1I = dloss1H * self.sigmoid_derivative(self.calc_layers['1I'])
        dlossw1 = self.train_targets.T.dot(dloss1I)
        dlossb1 = np.sum(dlossw1, axis=0)

        self.weight_bias['w1'] = self.weight_bias['w1'] - self.learning_rate * dlossw1
        self.weight_bias['w2'] = self.weight_bias['w2'] - self.learning_rate * dlossw2
        self.weight_bias['b1'] = self.weight_bias['b1'] - self.learning_rate * dlossb1
        self.weight_bias['b2'] = self.weight_bias['b2'] - self.learning_rate * dlossb2

    def fit(self):
        self.start_weights()  # initialize weights and bias

        for i in range(self.iterations):
            prediction, loss = self.forward_propagation()
            self.backward_propagation(prediction)
            print(loss," ",self.acc(self.train_targets,self.convert(prediction))," ",self.convert(prediction))
            self.loss.append(loss)




input_set = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1])
target_set = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
NN = NeuralNetwork(input=input_set, train_targets=target_set)
NN.fit()