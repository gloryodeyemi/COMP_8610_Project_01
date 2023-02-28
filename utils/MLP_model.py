from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout
from keras.constraints import MaxNorm
from keras.layers import BatchNormalization
from keras.optimizers import SGD

import matplotlib.pyplot as plt
import pickle

# standardize data
def standardize_data(data):
    print("Standardizing data...")
    print("---------------------")
    x_features = data.loc[:, data.columns != 'target']
    target_variable = data.loc[:, 'target']
    scaler = StandardScaler()
    scaler.fit(x_features)
    x_features = scaler.transform(x_features)
    return x_features, target_variable

# split dataset into input and target variables
def split_data(x_features, target_variable):
    print("Splitting dataset...")
    print("--------------------")
    # split data into training (80%) and testing (20%)
    x_train, x_test, y_train, y_test = train_test_split(x_features, target_variable, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test


# define the MLP model baseline
def build_model_baseline():
    print("Building model...")
    print("-----------------")
    model = Sequential([
        # dense layer 1
        Dense(23, input_shape=(23,), kernel_initializer = 'uniform', activation='relu'),

        # dense layer 2
        Dense(12, kernel_initializer = 'uniform', activation='relu'),

        # output layer
        Dense(1, kernel_initializer = 'uniform', activation='sigmoid'),
    ])
    return model

# define the MLP model with dropout
def build_model_dropout():
    print("Building model with dropout...")
    print("------------------------------")
    model = Sequential([
        # dense layer 1
        Dense(23, input_shape=(23,), kernel_initializer = 'uniform', activation='relu', kernel_constraint=MaxNorm(3)),
        Dropout(0.2),
        # dense layer 2
        Dense(23, kernel_initializer = 'uniform', activation='relu', kernel_constraint=MaxNorm(3)),
        Dropout(0.2),
        # output layer
        Dense(1, kernel_initializer = 'uniform', activation='sigmoid'),
    ])
    return model

# define the MLP model with batch normalization
def build_model_batch():
    print("Building model with batch normalization...")
    print("------------------------------------------")
    model = Sequential([
        # dense layer 1
        Dense(23, input_shape=(23,), kernel_initializer = 'uniform', activation='relu', kernel_constraint=MaxNorm(3)),
        Dropout(0.2),
        BatchNormalization(),
        # dense layer 2
        Dense(23, kernel_initializer = 'uniform', activation='relu', kernel_constraint=MaxNorm(3)),
        Dropout(0.2),
        BatchNormalization(),
        # output layer
        Dense(1, kernel_initializer = 'uniform', activation='sigmoid'),
    ])
    return model

# compile the model with adam optimizer
def compile_model_adam(model):
    print("Compiling the model with adam optimizer...")
    print("------------------------------------------")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
# compile the model with SGD optimizer
def compile_model_sdg(model):
    print("Compiling the model with SGD optimizer...")
    print("-----------------------------------------")
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

# train the model
def train_model(model, x_train, y_train):
    print("Training the model")
    print("------------------")
    history = model.fit(x_train, y_train, epochs=100, validation_split=0.2)
    return history

# save the trained model
def save_model(model, alias_):
    pickle.dump(model, open(f'model_{alias_}.pkl', 'wb'))

# accuracy results
def get_accuracy(model, x_test, y_test, x_train, y_train):    
    print("Results:")
    print("--------")
    train_scores = model.evaluate(x_train, y_train, verbose=False)
    print("Training Accuracy: %.2f%%\n" % (train_scores[1] * 100))
    print("Training loss: %.2f%%\n" % (train_scores[0] * 100))
    test_scores = model.evaluate(x_test, y_test, verbose=False)
    print("Testing accuracy: %.2f%%\n" % (test_scores[1] * 100))
    print("Testing loss: %.2f%%\n" % (test_scores[0] * 100))
    
def plot_accuracy(history, alias_):
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.legend()
    plt.savefig(f'history_{alias_}.png')
    plt.show()

# make predictions
def predict(model, x_test):
    y_pred = model.predict(x_test > 0.5).astype("int32")
    return y_pred

# confusion matrix
def con_matrix(actual, predicted, alias_):
    c_matrix = confusion_matrix(actual, predicted)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = c_matrix, display_labels = [0, 1])
    cm_display.plot()
    plt.savefig(f'confusion_matrix_{alias_}.png')
    plt.show()


# ROC curve
def roc_curve_draw(model, x_test, y_test, alias_):
    y_test_pred_probability = model.predict(x_test)
    FPR, TPR, _ = roc_curve(y_test, y_test_pred_probability)

    plt.plot(FPR, TPR)
    plt.plot([0,1], [0,1], '--', color='black')
    plt.title('ROC Curve')
    plt.xlabel('False-Positive Rate')
    plt.ylabel('True-Positive Rate')
    
    plt.savefig(f'roc_curve_{alias_}.png')
    plt.show()
    plt.clf()

