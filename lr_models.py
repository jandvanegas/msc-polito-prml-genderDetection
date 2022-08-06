import math

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import fmin_l_bfgs_b

import evaluation
import prml


def train_binary_regression(training, test, lambda_param, pi):
    (DTR, LTR), (DTE, LTE) = training, test
    log_reg_func = log_reg_func_factory(
        DTR, LTR, lambda_param=lambda_param, pi=pi)
    omegas, f, d = fmin_l_bfgs_b(log_reg_func,
                                 x0=np.zeros(DTR.shape[0] + 1),
                                 approx_grad=True,
                                 iprint=0)
    w = omegas[0:-1]
    b = omegas[-1]
    prediction = prml.vcol(w).T @ DTE + b
    score = (np.ones((1, DTE.shape[1])) * prediction > 0).astype(int)
    scores = 1 - np.sum(score == prml.vrow(LTE)) / DTE.shape[1]
    return w, b, scores, prediction


def log_reg_func_factory(DTR: np.ndarray,
                         LTR: np.ndarray,
                         lambda_param: float,
                         pi: float):
    Z = 2 * LTR - 1

    def log_reg_func(x):
        w = prml.vcol(x[0:-1])
        b = x[-1]
        regularization = np.power(np.linalg.norm(w), 2) * lambda_param / 2.0
        true_class = DTR[:, Z == 1]
        false_class = DTR[:, Z == -1]
        loss_true_class = pi * np.sum(
            np.logaddexp(0, - (w.T @ true_class + b))) / true_class.shape[1]
        loss_false_class = (1 - pi) * np.sum(
            np.logaddexp(0, + (w.T @ false_class + b))) / false_class.shape[1]
        return regularization + loss_true_class + loss_false_class

    return log_reg_func


def evaluate_lambdas(training, test, lambdas, Cfn, Cfp, prior, pi):
    min_dcfs = np.zeros(lambdas.shape[0])
    for i, lambda_value in enumerate(lambdas):
        _, _, _, prediction = train_binary_regression(
            training=training, test=test, lambda_param=lambda_value, pi=pi)
        min_dcfs[i] = evaluation.binary_evaluation(
            np.squeeze(prediction), test[1], prior, Cfn, Cfp)
    return min_dcfs


def get_dcfs_iterating_lambdas(training, test, lambdas, Cfn, Cfp, prior, pi):
    min_dcfs = evaluate_lambdas(training, test, lambdas, Cfn, Cfp, prior, pi)
    return lambdas, min_dcfs,


def cross_validate_lambdas(D, L, K, prior, pi):
    k_data = prml.split_data_k(D, L, k_number=K)
    lambdas = np.linspace(0, 1, num=20, endpoint=True)
    all_min_dcfs = np.empty((0, lambdas.shape[0]))
    for training, test in k_data:
        min_dcfs = evaluate_lambdas(
            training, test, lambdas, Cfn=1, Cfp=1, prior=prior, pi=pi)
        all_min_dcfs = np.append(all_min_dcfs, prml.vrow(min_dcfs), axis=0)
        best_lambda = lambdas[np.argmin(min_dcfs)]
        mindcf = np.min(min_dcfs)
        print(f"best lambda {best_lambda} with min_dcf {mindcf}")
    return lambdas, all_min_dcfs


def plot_cross_validated_lambdas(lambdas, all_min_dcfs, label):
    average_dcfs = np.average(all_min_dcfs, axis=0)
    print(f"Min lambda of average min dcfs {lambdas[np.argmin(average_dcfs)]}")
    plt.plot(lambdas, average_dcfs, label=label)


def cross_validate_linear_regression(D, L, K, lambda_val, prior, pi):
    k_data = prml.split_data_k(D, L, k_number=K)
    score_total = []
    for training, test in k_data:
        DTE, LTE = test
        _, _, score_1, prediction_class_1 = train_binary_regression(
            training=training, test=test, lambda_param=lambda_val, pi=pi)

        result = evaluation.binary_min_detection_cost(
            np.squeeze(prediction_class_1), LTE, prior, 1, 1)
        score_total.append(result)
    return np.average(np.array(score_total))


def run_priors_and_pi(D, L):
    pi_values = [0.1, 0.5, 0.6, 0.9]
    prior_values = [0.1, 0.5, 0.6, 0.9]
    results = {'pi': {}}
    for pi in pi_values:
        results['pi'][str(pi)] = {}
        for prior in prior_values:
            results['pi'][str(pi)][
                str(prior)] = cross_validate_linear_regression(D, L, K=5,
                                                               lambda_val=0.0,
                                                               prior=prior,
                                                               pi=pi)
    prml.pprint(results)


def iterate_test_results(Dtr, Ltr, DTe, LTe, label):
    pi_values = [0.1, 0.5, 0.9]
    prior_values = [0.1, 0.5, 0.9]
    print(f"Running {label}")
    for pi in pi_values:
        print(f"PiT {pi}")
        for prior in prior_values:
            run_over_test(Dtr, Ltr, DTe, LTe, pi, prior)


def run_over_test(Dtr, Ltr, DTe, LTe, pi, prior):
    log_reg_func = log_reg_func_factory(
        Dtr, Ltr, lambda_param=0.0, pi=pi)
    omegas, f, d = fmin_l_bfgs_b(
        log_reg_func, x0=np.zeros(Dtr.shape[0] + 1), approx_grad=True,
        iprint=0)
    w = omegas[0:-1]
    b = omegas[-1]
    prediction = prml.vcol(w).T @ DTe + b
    result = evaluation.binary_min_detection_cost(
        np.squeeze(prediction), LTe, prior, 1, 1)
    print(f'Prior {prior} getting \tmin dcf: {result}')
    return np.squeeze(prediction)

