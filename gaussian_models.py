import math

import evaluation
import prml
import numpy as np


def gaussian_classify_leave_one_out(D, L, classifier, pi):
    samples = D.shape[1]
    errors = []
    min_dcfs = []
    for i in range(samples):
        DTR = np.delete(D, i, axis=1)
        LTR = np.delete(L, i)
        _mean, _covariance, data_by_class = compute_gaussian_classifier(
            DTR, LTR, classifier=classifier, classes=(1, 0))
        DTE = D[:, i:i + 1]
        LTE = L[i:i + 1]
        prior = np.array(
            [[pi],
             [1 - pi]])

        scores = get_score_matrix(DTE, _mean, _covariance,
                                  logarithmically=False)
        eval = prml.evaluate(DTE, LTE, scores, prior)
        errors.append(eval['err'])
        res = evaluation.binary_min_detection_cost(
            scores, L, pi, Cfn=1, Cfp=1)
        min_dcfs.append(res)
    return np.average(errors), np.average(min_dcfs)


def gaussian_classify_k_cross_validation(D, L, classifier, pi, K):
    k_data = prml.split_data_k(D, L, k_number=K)
    errors = []
    min_dcfs = []
    for training, test in k_data:
        DTR, LTR = training
        DTE, LTE = test
        _mean, _covariance, data_by_class = compute_gaussian_classifier(
            DTR, LTR, classifier=classifier, classes=(0, 1))
        prior = np.array([[1 - pi], [pi]])

        scores = get_score_matrix(DTE, _mean, _covariance,
                                  logarithmically=False)
        eval = prml.evaluate(DTE, LTE, scores, prior)
        errors.append(eval['err'])
        ll = np.log(eval['posterior'][1] / (eval['posterior'][0]))
        res = evaluation.binary_min_detection_cost(ll, LTE, pi, Cfn=1, Cfp=1)
        min_dcfs.append(res)
    return np.average(errors), np.average(min_dcfs)


def compute_gaussian_classifier(DTR, LTR, classifier, classes):
    data_by_classes = split_by_classes(DTR, LTR, classes)
    mean_by_class, covariance_by_class = get_mean_and_covariance(
        data_by_classes)
    classifier_covariance = compute_classifier_covariance(covariance_by_class,
                                                          classifier, DTR, LTR)
    return mean_by_class, classifier_covariance, data_by_classes


def split_by_classes(data, labels, classes):
    result = []
    for _class in classes:
        result.append(data[:, labels == _class])
    return result


def compute_classifier_covariance(covariance, classifier, D, L):
    if classifier == 'mvg':
        return covariance
    elif classifier == 'naive':
        return diagonalize_covariance(covariance)
    elif classifier == 'tied_gaussian':
        covariance = prml.covariance_within(D, L)
        return [covariance for _ in range(2)]
    elif classifier == 'tied_naive':
        covariance = prml.covariance_within(D, L)
        return diagonalize_covariance([covariance for _ in range(2)])


def diagonalize_covariance(covariance_by_class):
    diagonalized_covariances = []
    for covariance in covariance_by_class:
        size = covariance.shape[0]
        diagonalized_covariances.append(covariance * np.identity(size))
    return diagonalized_covariances


def get_mean_and_covariance(data_by_classes):
    mean = []
    covariance = []
    for class_data in data_by_classes:
        mean.append(prml.mean(class_data))
        covariance.append(prml.covariance(class_data))
    return mean, covariance


def get_score_matrix(samples, mean_by_class, covariance_by_class,
                     logarithmically=False):
    samples_number = samples.shape[1]
    score = np.empty((0, samples_number))
    for mean, covariance in zip(mean_by_class, covariance_by_class):
        class_score = logpdf_GAU_ND(samples, mean, covariance)
        if not logarithmically:
            class_score = np.exp(class_score)
        score = np.vstack([score, class_score])
    return score


def logpdf_GAU_ND(X, mu, C):
    """
    Computes the Multivariate Gaussian Density

    :param X: matrix features x samples
    :param mu: mean
    :param C: empirical covariance
    :return:
    """
    M = X.shape[0]
    first_term = - 0.5 * M * np.log(2 * math.pi)
    centered_x = X - mu
    second_term = - 0.5 * np.linalg.slogdet(C)[1]
    third_term = - 0.5 * np.sum(
        (centered_x.T @ np.linalg.inv(C)) * centered_x.T,
        axis=1)
    return first_term + second_term + third_term


def Outliers_IQR(x, k):
    """
    Compute the outliers based on IQR
    :param x: matrix of features
    :return: np array with the index of the outliers
    """
    q1 = np.percentile(x, 25, axis=1)
    q3 = np.percentile(x, 75, axis=1)
    IQR = q3 - q1
    outMayor = q3 + (k * IQR)
    outMenor = q1 - (k * IQR)

    outliers = []
    count = 0
    for j in range(0, x.shape[1]):
        for i in range(0, x.shape[0]):
            if x[i, j] > outMayor[i] or x[i, j] < outMenor[i]:
                outliers.append(j)
                count += 1
    return np.unique(outliers)


def try_all_methods(D_formats, L, pi, K):
    methods = ['mvg', 'naive', 'tied_gaussian', 'tied_naive']
    for key, _D in D_formats.items():
        for method in methods:
            error, min_dcf = gaussian_classify_k_cross_validation(
                _D, L, method, pi, K)
            print(
                f"{error} \t error {min_dcf} min_dcf applying \t\t "
                f"Method {method} in data \t{key}")


def test_result(DTr, LTr, DTe, LTe, pi, method):
    _mean, _covariance, _ = compute_gaussian_classifier(
        DTr, LTr, classifier=method, classes=(0, 1))
    prior = np.array([[1 - pi], [pi]])

    scores = get_score_matrix(DTe, _mean, _covariance)
    eval = prml.evaluate(DTe, LTe, scores, prior)
    ll = np.log(eval['posterior'][1] / (eval['posterior'][0]))
    return evaluation.binary_min_detection_cost(ll, LTe, pi, Cfn=1, Cfp=1)


def iterate_test_results(DTr, LTr, DTe, LTe, label):
    methods = ['mvg', 'naive', 'tied_gaussian', 'tied_naive']
    pis = [0.1, 0.5, 0.9]
    print(f"Running: {label}")
    for method in methods:
        print(f"Method: {method}")
        for pi in pis:
            result = test_result(DTr, LTr, DTe, LTe, pi, method)
            print(f'Pi {pi} getting \tmin dcf: {result}')
