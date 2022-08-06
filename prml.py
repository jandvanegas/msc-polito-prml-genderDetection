"""
Authors: Victor, Juan Vanegas
Pattern Recognition and Machine Learning Project to recognize quality of wines.

Conditions
Python Version: 3.8
"""
import json
import logging
import math
import typing
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import scipy
import scipy.linalg
import scipy.special

PLOT_MAX_COLUMNS = 2

logging.basicConfig(filename='prml.log', level=logging.DEBUG)

hFea = {
    0: 'Fixed acidity',
    1: 'Volatile acidity',
    2: 'Citric acid',
    3: 'Residual sugar',
    4: 'Chlorides',
    5: 'Free sulfur dioxide',
    6: 'Total sulfur dioxide',
    7: 'Density',
    8: 'pH',
    9: 'Sulphates',
    10: 'Alcohol'
}

pcaFeat = {
    0: 'Feature 0',
    1: 'Feature 1',
    2: 'Feature 2',
    3: 'Feature 3',
    4: 'Feature 4',
    5: 'Feature 5',
    6: 'Feature 6',
    7: 'Feature 7',
    8: 'Feature 8',
}


def main():
    load('./data/Train.txt', features=11)


def load(file_name: str, features: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads a file in csv format and parse it into two numpy arrays
    one with the features columns and the second one with the labels.

    :param file_name: str
    :param features: int
    :return: numpy arrays: D: features x n, Labels: n,
    """
    DList = []
    labelsList = []
    with open(file_name) as f:
        for line in f:
            try:
                attrs = line.split(',')[0:features]
                attrs = vcol(np.array([float(i) for i in attrs]))
                label = line.split(',')[-1].strip()
                DList.append(attrs)
                labelsList.append(label)
            except Exception as ex:
                logging.error(ex)

    D = np.hstack(DList)
    L = np.array(labelsList, dtype=np.int32)
    logging.info(f"Loaded D with shape {D.shape}")
    logging.info(f"Loaded L with shape {L.shape}")
    return D, L


def vcol(vector: np.ndarray):
    """
    Converts vector into column vector
    :param vector: one dimensional vector np.ndarray (size,)
    :return: vector: two dimensional vector (size, 1)
    """
    return vector.reshape((vector.size, 1))


def plot_D(D: np.ndarray, L, feature_names: typing.Dict):
    feature_number = len(feature_names.keys())
    number_rows = math.ceil(feature_number / PLOT_MAX_COLUMNS)
    number_columns = (
        feature_number if feature_number < PLOT_MAX_COLUMNS
        else PLOT_MAX_COLUMNS)
    fig, axes = plt.subplots(number_rows, number_columns)
    for key, value in feature_names.items():
        x = math.floor(key / PLOT_MAX_COLUMNS)
        y = key % PLOT_MAX_COLUMNS
        axes[x, y].hist(D[key, L == 0], alpha=0.8, label="Low")
        axes[x, y].hist(D[key, L == 1], alpha=0.8, label="High")
        axes[x, y].legend(loc='upper right', prop={'size': 18})
        axes[x, y].set_title(value, fontsize=18)
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=1.5,
                        top=4,
                        wspace=0.4,
                        hspace=0.4)
    plt.show()


def plot_in_two_parts_feature(D: np.ndarray, L, feature_number: int,
                              threshold):
    fig, axes = plt.subplots(1, 2)
    axes[0].hist(D[feature_number, (L == 0) & (D[feature_number] < threshold)],
                 alpha=0.8,
                 label="Low")
    axes[0].hist(D[feature_number, (L == 1) & (D[feature_number] < threshold)],
                 alpha=0.8,
                 label="High")
    axes[0].legend(loc='upper right', prop={'size': 18})
    axes[0].set_title(f"{hFea[feature_number]} < {threshold}", fontsize=18)

    axes[1].hist(
        D[feature_number, (L == 0) & (D[feature_number] >= threshold)],
        alpha=0.8,
        label="Low")
    axes[1].hist(
        D[feature_number, (L == 1) & (D[feature_number] >= threshold)],
        alpha=0.8,
        label="High")
    axes[1].legend(loc='upper right', prop={'size': 18})
    axes[1].set_title(f"{hFea[feature_number]} >= {threshold}", fontsize=18)
    config = {"left": 0.1,
              "bottom": 0.1,
              "right": 1.5,
              "top": 1,
              "wspace": 0.4,
              "hspace": 0.4}
    plt.subplots_adjust(**config)
    plt.show()


def mean(matrix_d):
    """
    Compute mean of a matrix_d
    :param matrix_d: features x N data matrix
    :return: features x 1 column vector
    """
    return vcol(matrix_d.mean(1))


def covariance(matrix_d):
    """
    Compute covariance of a np.ndarray features x N
    :param matrix_d: features x N
    :return: covariance matrix : features x features
    """
    N = matrix_d.shape[1]
    mu = mean(matrix_d)
    data_centered = (matrix_d - mu)
    return data_centered @ data_centered.T / N


def z_score(matrix_d):
    mean_value = mean(matrix_d)
    std = np.diag(covariance(matrix_d))
    return np.divide((matrix_d - mean_value), vcol(std))


def scale(matrix_d):
    minimun = vcol(np.min(matrix_d, axis=1))
    maximun = vcol(np.max(matrix_d, axis=1))
    return (matrix_d - minimun) / (maximun - minimun)


def gaussianization(D):
    temp = D.argsort(axis=1)
    rank = (temp.argsort(axis=1) + 1) / (D.shape[1] + 2)
    return norm.ppf(q=rank), rank


def apply_test_gaussianization(D, Test):
    RankedTest = np.empty((D.shape[0], 0))
    for column in range(Test.shape[1]):
        rank = (np.sum(D < Test[:, column:column + 1], axis=1, keepdims=True) + 1) / (
                    D.shape[1] + 3)
        RankedTest = np.append(RankedTest, rank, axis=1)
    return norm.ppf(q=RankedTest)


def plot_pearson_correlation(D):
    pearson = covariance(D)
    for i in range(D.shape[0]):
        for j in range(D.shape[0]):
            pearson[i, j] = pearson[i, j] / (pearson[i, i] * pearson[j, j])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(pearson, interpolation='nearest', cmap='Blues')
    fig.colorbar(cax)
    plt.show()


def compute_pca(matrix_d, m):
    """
    Computes the Principal component Analysis of a Matrix
    :param matrix_d: (np.ndarray features, N) matrix
    :param m: number of components
    :return: Data projected over the m principal components
    """
    d_covariance = covariance(matrix_d)
    eig_values, eig_vectors = eigh(d_covariance)
    P = eig_vectors[:, 0:m]  # Eigen vectors to project
    DP = P.T @ matrix_d  # Matrix D projected on P
    return DP


def eigh(matrix_d):
    """
    Return eigen values and vectors using np. linalg.eigh but in desc order
    :param matrix_d: symmetric matrix
    :return: np.ndarray with eigen values, np.ndarray with eigen vectors
    """
    eig_values, eig_vectors = np.linalg.eigh(matrix_d)
    return eig_values[::-1], eig_vectors[:, ::-1]


def compute_accumulated_variance(D):
    return np.trace(covariance(D))


def plot_accumulated_variance_ratio(D):
    number_of_features = D.shape[0]
    accumulated_variance_ratio = np.zeros(number_of_features)
    M = np.linspace(1, number_of_features, number_of_features, endpoint=True)
    acc_var_D = compute_accumulated_variance(D)
    for m in range(number_of_features):
        DP_pca = compute_pca(D, m=m + 1)
        accumulated_variance_ratio[m] = compute_accumulated_variance(
            DP_pca) / acc_var_D
    plt.plot(M, accumulated_variance_ratio, marker='o')
    plt.plot((1, 11), (0.95, 0.95), 'k-')
    plt.xlabel('Number of eigen vector m')
    plt.ylabel('Accumulated variance ratio')
    plt.show()


def vrow(vector: np.ndarray):
    """
    Converts vector into row vector
    :param vector: one dimensional vector np.ndarray (size,)
    :return: vector: two dimensional vector (1, size)
    """
    return vector.reshape((1, vector.size))


def covariance_between(matrix_d, matrix_l):
    """
    Return covariance between classes
    :param matrix_d: features x N
    :param matrix_l: classes vector
    :return: covariance matrix features x features
    """
    classes = set(matrix_l)
    features = matrix_d.shape[0]
    N = matrix_d.shape[1]
    s_b = np.zeros((features, features))
    mu = mean(matrix_d)
    for class_l in classes:
        d_class = matrix_d[:, matrix_l == class_l]
        nc = d_class.shape[1]
        mu_c = mean(d_class)
        classes_distance = mu_c - mu
        summation = np.multiply(nc, classes_distance) @ classes_distance.T
        s_b = s_b + summation
    return s_b / N


def covariance_within(matrix_d, matrix_l):
    classes = set(matrix_l)
    N = matrix_d.shape[1]
    features = matrix_d.shape[0]
    s_w = np.zeros((features, features))
    for class_l in classes:
        d_class = matrix_d[:, matrix_l == class_l]
        mu_c = mean(d_class)
        central_data = d_class - mu_c
        class_summation = central_data @ central_data.T
        s_w = s_w + class_summation
    return s_w / N


def split_data(D: np.ndarray, L: np.ndarray, proportion, seed=0):
    nTrain = int(D.shape[1] * proportion)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]

    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)


def split_data_k(D: np.ndarray, L: np.ndarray, k_number, seed=0):
    np.random.seed(seed)
    indexes = np.linspace(0, D.shape[1], num=k_number + 1, dtype=int)
    k_parts = []
    shuffles_indexes = np.random.permutation(D.shape[1])
    for k in range(k_number):
        k_indexes = shuffles_indexes[indexes[k]: indexes[k + 1]]
        k_parts.append((D[:, k_indexes], L[k_indexes]))

    training_test_data = []
    for k_test in range(k_number):
        test = k_parts[k_test]
        training_D = np.empty((D.shape[0], 0))
        training_L = np.empty((0, 0))
        for k_training in range(k_number):
            if k_test != k_training:
                training_D = np.append(training_D, k_parts[k_training][0],
                                       axis=1)
                training_L = np.append(training_L, k_parts[k_training][1])
        training_test_data.append(((training_D, training_L), test))
    return training_test_data


def optimal_bayes(ll_ratio, L, prior, Cfn, Cfp, threshold=None):
    """
    Computes optimal bayes decision
    :param ll_ratio: loglikelihood ratio
    :param L: real labels
    :param prior: pi value of prior probability
    :param Cfn: cost of false negatives
    :param Cfp: cost of false positives
    :param threshold: [optional] decision bayes threshold
    :return:
    """
    if threshold is None:
        threshold = - np.log((prior * Cfn) / ((1 - prior) * Cfp))
    prediction = ll_ratio > threshold
    confusion = compute_confusion_matrix(np.unique(L), L, prediction)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FN = confusion[0, 1]
    FP = confusion[1, 0]
    FPR = FP / (FP + TN)
    FNR = FN / (FN + TP)
    return prediction, FPR, FNR, confusion


def compute_confusion_matrix(labels, L_real, L_predicted):
    confusion = np.zeros((len(labels), len(labels)))
    for column in labels:
        current = L_predicted[L_real == column]
        for row in labels:
            confusion[row, column] = np.count_nonzero(current == row)
    return confusion


def compute_join(score, class_probability, logarithmically=False):
    if logarithmically:
        return score + np.log(class_probability)
    return class_probability * score


def compute_marginal(SJoint, logarithmically=False):
    if logarithmically:
        return vrow(scipy.special.logsumexp(SJoint, axis=0))
    return vrow(SJoint.sum(0))


def compute_posterior(SJoint, SMarginal, logarithmically=False):
    if logarithmically:
        return np.exp(SJoint - SMarginal)
    return SJoint / SMarginal


def evaluate(DTE, LTE, scores, prior, logarithmically=False):
    SJoint = compute_join(scores, prior, logarithmically=logarithmically)
    SMarginal = compute_marginal(SJoint, logarithmically=logarithmically)
    posterior = compute_posterior(SJoint, SMarginal,
                                  logarithmically=logarithmically)
    SPost = np.argmax(posterior, axis=0, keepdims=False)
    accuracy = np.sum(SPost == LTE) / DTE.shape[1]
    err = 1.0 - accuracy
    return {
        'Sjoint': SJoint,
        'SMarginal': SMarginal,
        'posterior': posterior,
        'SPost': SPost,
        'acc': accuracy,
        'err': err
    }


def pprint(value):
    if type(value) == 'str':
        value = json.loads(value)
    print(json.dumps(value, indent=4, sort_keys=True))
