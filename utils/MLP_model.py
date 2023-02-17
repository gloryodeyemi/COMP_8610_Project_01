from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn import metrics

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib

# split dataset into input and target variables
def split_data(data):
    print("Splitting dataset...")
    print("--------------------")
    x_features = data.loc[:, data.columns != 'target']
    target_variable = data.loc[:, 'target']
    # split data into training (80%) and testing (20%)
    x_train, x_test, y_train, y_test = train_test_split(x_features, target_variable, test_size=0.2, random_state=42)
    # split training split into training (80%) and validation (20%)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test, x_val, y_val


# define the MLP model
def build_model():
    print("Building model...")
    print("-----------------")
    model = Sequential([
        # dense layer 1
        Dense(32, input_shape=(23,), activation='relu'),

        # dense layer 2
        Dense(16, activation='relu'),

        # output layer
        Dense(1, activation='sigmoid'),
    ])
    return model


# compile the model
def compile_model(model):
    print("Compiling the model...")
    print("----------------------")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# train the model
def train_model(model, x_train, y_train):
    print("Training the model")
    print("------------------")
    model.fit(x_train, y_train, epochs=100)
    return model

# accuracy results
def get_accuracy(model, x_test, y_test):    
    print("Results:")
    print("--------")
    scores = model.evaluate(x_test, y_test, verbose=False)
    print("Testing accuracy: %.2f%%\n" % (scores[1] * 100))
    print("Testing loss: %.2f%%\n" % (scores[0] * 100))


# make predictions
def predict(model, x_test):
    y_pred = model.predict(x_test > 0.5).astype("int32")
    return y_pred


# confusion matrix
def con_matrix(actual, predicted):
    c_matrix = confusion_matrix(actual, predicted)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = c_matrix, display_labels = [0, 1])
    cm_display.plot()
    plt.savefig('images/confusion_matrix.png')
    plt.show()


# ROC curve
def roc_curve_draw(model, x_test, y_test):
    y_test_pred_probability = model.predict(x_test)
    FPR, TPR, _ = roc_curve(y_test, y_test_pred_probability)

    plt.plot(FPR, TPR)
    plt.plot([0,1], [0,1], '--', color='black')
    plt.title('ROC Curve')
    plt.xlabel('False-Positive Rate')
    plt.ylabel('True-Positive Rate')
    
    plt.savefig('images/roc_curve.png')
    plt.show()
    plt.clf()

