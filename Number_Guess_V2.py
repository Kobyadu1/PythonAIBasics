# 15 - 30 - 30 - 10
# middle layer is a leaky relu followed by relu with sigmoid on the output
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import keras
import tensorflow as tf
from keras.datasets import mnist
from keras import layers
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
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
        if epoch%100==0:
            # Clear the previous plot
            N = np.arange(0, len(self.losses))

            # You can chose the style of your preference
            # print(plt.style.available) to see the available options
            plt.style.use("seaborn")

            # Plot train loss, train acc, val loss and val acc against epochs passed
            plt.figure()
            plt.plot(N, self.losses, label="train_loss")
            plt.plot(N, self.acc, label="train_acc")
            plt.plot(N, self.val_losses, label="val_loss")
            plt.plot(N, self.val_acc, label="val_acc")
            plt.title("Training Loss and Accuracy [Epoch {}]".format(epoch))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            plt.show()


class list_holder():
    temp_list = []
    
    def __init__(self,temp_list):
        self.temp_list = temp_list


def convert(self, prediction):
    max = np.amax(prediction)
    prediction2 = []
    for i in prediction[0]:
        if i == max:
            prediction2.append(1)
        else:
            prediction2.append(0)
    return prediction2, np.argmax(prediction2)

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


# Turning matrix above into np arrays
X = np.array(input_set)
Y = np.array(target)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0, test_size=.24)
inputs = keras.Input(shape=(15, ))
x = layers.Dense(30, activation="relu")(inputs)
x = layers.Dense(30)(x)
x = keras.layers.LeakyReLU(alpha=0.01)(x)
output = layers.Dense(10, activation="sigmoid")(x)

# model creation
model = keras.Model(inputs=inputs, outputs=output)
print(model.summary())

# Setting certain paras for model
model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=False),
              optimizer=keras.optimizers.Adam(learning_rate=.0005),
              metrics=['accuracy'])

# validation_split=0.2 is the float that - epoch = iterations
# Training
plot_losses = TrainingPlot()
# validation_split=.25
history = model.fit(X_train, Y_train, batch_size=32, epochs=400, verbose=0, validation_data=(X_test, Y_test),
                    callbacks=[plot_losses])

# Test - verbose = 0 is nothing, 1 is progress bar, 2 is display
test_scores = model.evaluate(X, Y, verbose=0)
print("Full Loss:", test_scores[0])
print("Full Accuracy:", test_scores[1], '\n')

test_scores = model.evaluate(X_train, Y_train, verbose=0)
print("Train Loss:", test_scores[0])
print("Train Accuracy:", test_scores[1], '\n')

test_scores = model.evaluate(X_test, Y_test, verbose=0)
print("Test Loss:", test_scores[0])
print("Test Accuracy:", test_scores[1], '\n')

# Saving the model
#model.save("path_to_my_model")
#del model
# Recreate the exact same model purely from the file:
#model = keras.models.load_model("path_to_my_model")

# input_Array = [1,1,1,1,0,1,1,1,1,1,0,1,1,1,1]
input_Array = [0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1]  # 3
input_Array = tf.convert_to_tensor(input_Array)
tf.reshape(input_Array, (15,))
pred = model(input_Array)

print(np.argmax(pred))