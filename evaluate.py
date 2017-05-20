#from __future__ import print_function

# Authors: Yann N. Dauphin, Vlad Niculae, Gabriel Synnaeve
# License: BSD

import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline

import data_engine

def load_data():
    engine = data_engine.data_engine()
    engine.load()
    return [engine.train_x.reshape([-1, 32 * 32]), engine.train_y, engine.test_x.reshape([-1, 32 * 32]), engine.test_y]

def get_prediction(X_train, Y_train, X_test, Y_test):
    logistic = linear_model.LogisticRegression()
    rbm = BernoulliRBM(random_state=0, verbose=True)

    classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])

    rbm.learning_rate = 0.005
    rbm.n_iter = 100
    # More components tend to give better prediction performance, but larger
    # fitting time
    rbm.n_components = 64
    logistic.C = 6000.0

    # Training RBM-Logistic Pipeline
    classifier.fit(X_train, Y_train)

    # Training Logistic regression
    logistic_classifier = linear_model.LogisticRegression(C=100.0)
    logistic_classifier.fit(X_train, Y_train)

    print("Logistic regression using RBM features:\n%s\n" % (
        metrics.classification_report(
            Y_test,
            classifier.predict(X_test))))

    print("Logistic regression using raw pixel features:\n%s\n" % (
        metrics.classification_report(
            Y_test,
            logistic_classifier.predict(X_test))))

X_train, Y_train, X_test, Y_test = load_data()

get_prediction(X_train, Y_train, X_test, Y_test)