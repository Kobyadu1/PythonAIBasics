# 15 - 12 - 10 - 10
# Middle layers are leaky relu with softmax on output layer

import numpy as np
import keras
import tensorflow as tf
from keras.datasets import mnist
from keras import layers
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from keras.layers.advanced_activations import LeakyReLU, PReLU
# from PIL import Image

# Image loading will do this later - add a function
#im = Image.open("image.png")
#pic = im.load()
# pix[3,3]

#static setting of "train" "test" but will be revised
#target should only exist for train data


# Sourced from https://github.com/kapil-varshney/utilities/blob/master/training_plot/training_plot_ex_with_cifar10.ipynb
class TrainingPlot(keras.callbacks.Callback):

    # This function is called when the training begins
    def on_train_begin(self, logs={}):
        # Initialize the lists for holding the logs, losses and accuracies
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []

    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))

        # Before plotting ensure at least 2 epochs have passed
        if epoch%1000==0:
            # Clear the previous plot
            N = np.arange(0, len(self.losses))

            # You can chose the style of your preference
            # print(plt.style.available) to see the available options
            plt.style.use("seaborn")

            # Plot train loss, train acc, val loss and val acc against epochs passed
            plt.figure()
            plt.plot(N, self.losses, label="train_loss")
            plt.plot(N, self.acc, label="train_acc")
            print(epoch," ", self.acc)
            print(epoch," ",self.val_acc)
            plt.plot(N, self.val_losses, label="val_loss")
            plt.plot(N, self.val_acc, label="val_acc")
            plt.title("Training Loss and Accuracy [Epoch {}]".format(epoch))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
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

number_input = (input('Enter a number: '))
iterations = int(input('Enter number of iterations: '))
# size_of_second = int(input('Enter the size of second layer: '))
trial_number = int(input('Enter number of model trials (creations): '))

X = np.array(input_set)
Y = np.array(target)


inputs = keras.Input(shape=(15, ))
x = layers.Dense(12)(inputs)
x = keras.layers.LeakyReLU(alpha=0.01)(x)
x = layers.Dense(10)(x)
x = keras.layers.LeakyReLU(alpha=0.01)(x)
output = layers.Dense(10, activation="softmax")(x)

model_dict = {}
data_dict = {}


for i in range(trial_number):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=i, test_size=.24)
    model = keras.Model(inputs=inputs, outputs=output, name="Number Guess Model " + str(i))
    model.compile(loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                  optimizer=keras.optimizers.SGD(learning_rate=.01),
                  metrics=['accuracy'])

    plot_losses = TrainingPlot()
   # history = model.fit(X, Y, batch_size=32, epochs=1001, verbose=0, validation_data=(X_test, Y_test),
   #                     callbacks=[plot_losses])
    history = model.fit(X, Y, batch_size=32, epochs=iterations, verbose=0, validation_data=(X_test, Y_test),
                        shuffle=True)

    acc_list = []
    test_scores = model.evaluate(X, Y, verbose=0)
    # print("Full Loss:", test_scores[0])
    # print("Full Accuracy:", test_scores[1], '\n')
    acc_list.append(test_scores[1])

    test_scores = model.evaluate(X_train, Y_train, verbose=0)
    # print("Train Loss:", test_scores[0])
    # print("Train Accuracy:", test_scores[1], '\n')
    acc_list.append(test_scores[1])

    test_scores = model.evaluate(X_test, Y_test, verbose=0)
    # print("Test Loss:", test_scores[0])
    # print("Test Accuracy:", test_scores[1], '\n')
    acc_list.append(test_scores[1])

    avg_acc = np.average([acc_list])
    print(avg_acc)
    data_dict[model] = [[X_train, X_test], [Y_train, Y_test]]
    model_dict[model] = avg_acc


index = np.amax(list(model_dict.values()))
model = None
for x in list(model_dict.keys()):
    if model_dict[x] == index:
        model = x

X_train = data_dict[model][0][0]
X_test = data_dict[model][0][1]
Y_train = data_dict[model][1][0]
Y_test = data_dict[model][1][1]

return_number = []
for i in number_input:
    if i == 9:
        input_list = input_set[33]
    else:
        input_list = input_set[int(i)*3]
    input_list = tf.convert_to_tensor(input_list, dtype=tf.int32)
    tf.reshape(input_list, (15,))
    return_number.append(np.argmax(model.predict(np.array([input_list, ]))))


print('The guessed number is: ', return_number)
test_scores = model.evaluate(X, Y, verbose=0)
print("Full Loss:", test_scores[0])
print("Full Accuracy:", test_scores[1], '\n')
test_scores = model.evaluate(X_train, Y_train, verbose=0)
print("Train Loss:", test_scores[0])
print("Train Accuracy:", test_scores[1], '\n')
test_scores = model.evaluate(X_test, Y_test, verbose=0)
print("Test Loss:", test_scores[0])
print("Test Accuracy:", test_scores[1], '\n')




