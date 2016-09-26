#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import sys
import random
import math
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split

"""
AdaBoostの実装と「digits」での評価
"""

def adaboost(sample, weakLearner, rounds):
    distribution = normalize([1.] * len(sample))
    hypothesis = [None] * rounds
    alpha = [0] * rounds

    numEpochs = [None]
    errorList = [None]

    for i in range(rounds):
        def drawSample():
            return sample[draw(distribution)]
        hypothesis[i] = weakLearner(drawSample)
        hypothesisResult, error = errorCalc(hypothesis[i], sample, distribution)
        alpha[i] = 0.5 * math.log((1 - error) * 1.0 / (.0001 + error))
        distribution = normalize([d * math.exp(-alpha[i] * h) for (d, h) in zip(distribution, hypothesisResult)])
        print("Round %d, error %.3f" % (i + 1, error))
        numEpochs.append(i+1)
        errorList.append(error)

    numEpochs = np.array(numEpochs)
    errorList = np.array(errorList)
    plt.plot(numEpochs, errorList, '-o')
    plt.grid()
    plt.title('Error for each weak learner')
    plt.xlabel('Weak learner No.')
    plt.ylabel('Error')
    plt.show()
    #plt.savefig("error_rate_per_learner.png")
    #plt.clf()

    def totalHypothesis(x):
        return sign(sum(a * h(x) for (a, h) in zip(alpha, hypothesis)))
    return totalHypothesis

def errorCalc(h, samples, weights=None):
    if weights is None:
        weights = [1.] * len(samples)
    hypothesisResult = [h(x)*y for (x, y) in samples]
    return hypothesisResult, sum(a for (b, a) in zip(hypothesisResult, weights) if b < 0)

class Stump:
    def __init__(self):
        self.aboveLabel = None
        self.belowLabel = None
        self.branchThrehold = None
        self.branchFeature = None

    def classification(self, sets):
        if sets[self.branchFeature] >= self.branchThreshold:
            return self.aboveLabel
        else:
            return self.belowLabel

    def __call__(self, sets):
        return self.classification(sets)

def majorityVote(data):
    labels = [label for (st, label) in data]
    try:
        return max(set(labels), key=labels.count)
    except:
        return -1

def minimumError(data, h):
    positiveData, negativeData = ([(x, y) for (x, y) in data if h(x) == 1], [(x, y) for (x, y) in data if h(x) == -1])
    positiveError = sum(y == -1 for (x, y) in positiveData) + sum(y == 1 for(x, y) in negativeData)
    negativeError = sum(y == 1 for (x, y) in positiveData) + sum(y == -1 for(x, y) in negativeData)
    return min(positiveError, negativeError) * 1.0 / len(data)

def defaultError(data, h):
    return minimumError(data, h)

def bestThreshold(data, feature, errorFunction):
    thresholds = [sets[feature] for (sets, label) in data]
    def makeThreshold(th):
        return lambda x: 1 if x[feature] >= th else -1
    errors = [(threshold, errorFunction(data, makeThreshold(threshold))) for threshold in thresholds]
    return min(errors, key=lambda p: p[1])

def buildDecisionStump(drawSample, errorFunction=defaultError):
    data = [drawSample() for _ in range(500)]
    bestThresholds = [(i,) + bestThreshold(data, i, errorFunction) for i in range(len(data[0][0]))]
    feature, thre, _ = min(bestThresholds, key=lambda p: p[2])

    stump = Stump()
    stump.branchFeature = feature
    stump.branchThreshold = thre
    stump.aboveLabel = majorityVote([x for x in data if x[0][feature] >= thre])
    stump.belowLabel = majorityVote([x for x in data if x[0][feature] < thre])

    return stump

def draw(weight):
    choose = random.uniform(0, sum(weight))
    chooseIndex = 0

    for wt in weight:
        choose -= wt
        if choose <= 0:
            return chooseIndex
        chooseIndex += 1

def normalize(weight):
    norm = sum(weight)
    return tuple(m * 1.0/ norm for m in weight)

def sign(x):
    return 1 if x >= 0 else -1

def errorRate(h, data):
    return sum(1 for x, y in data if h(x) != y) * 1.0 / len(data)

def loadExample():
    train = [((1, 0, 1, 1), 1), ((1, 1, 1, 1), 1), ((1, 1, 1, 0), -1), ((1, 0, 1, 0), -1), ((1, 1, 0, 1), -1), ((1, 0, 0, 1), -1), ((1, 1, 0, 1), -1), ((0, 0, 1, 1), 1), ((0, 1, 0, 0), -1)]
    test = [((0, 1, 0, 1), 1), ((0, 0, 1, 0), -1), ((1, 0, 1, 1), 1)]
    return train, test

if __name__ == '__main__':

    #train, test = loadExample()
    train = []
    test = []
    digits = load_digits(2)
    data_train, data_test, label_train, label_test = train_test_split(digits.data, digits.target, train_size=0.75, random_state=2016)
    for i in range(len(label_train)):
        if label_train[i] != 1:
            label_train[i] = -1
        train.append((data_train[i], label_train[i]))
    for i in range(len(label_test)):
        if label_test[i] != 1:
            label_test[i] = -1
        test.append((data_test[i], label_test[i]))

    weakLearner = buildDecisionStump

    numLearners = []
    trainError = []
    testError = []
    for i in range(2):#6
        rounds = i*5#i
        h = adaboost(train, weakLearner, rounds)
        print("Training Error : %G" % errorRate(h, train))
        print("Test Error : %G" % errorRate(h, test))
        numLearners.append(rounds)
        trainError.append(errorRate(h, train))
        testError.append(errorRate(h, test))

    numLearners = np.array(numLearners)
    trainError = np.array(trainError)
    testError = np.array(testError)
    plt.plot(numLearners, trainError, '-o')
    plt.grid()
    plt.title('Error rate by number of learners(train)')
    plt.xlabel('Number of learners')
    plt.ylabel('Error rate')
    plt.show()

    plt.plot(numLearners, testError, '-o')
    plt.grid()
    plt.title('Error rate by number of learners(test)')
    plt.xlabel('Number of learners')
    plt.ylabel('Error rate')
    plt.show()
