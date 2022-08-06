import numpy as np
from matplotlib import pyplot as plt

import prml


def main_binary_analysis(score, L, Cfn, Cfp):
    # binary_evaluation(score, L, prior, Cfn, Cfp)
    # binary_min_detection_cost(score, L, prior, Cfn, Cfp)
    binary_plot_roc(score, L)
    binary_plot_error(score, L, colors=('y', 'b'), Cfn=Cfn, Cfp=Cfp)
    plt.show()


def binary_evaluation(ll_ratio, L, prior, Cfn, Cfp, t=None):
    _, FPR, FNR, _ = prml.optimal_bayes(ll_ratio, L, prior, Cfn, Cfp, t)
    DCFu = prior * Cfn * FNR + (1 - prior) * Cfp * FPR
    B_dummy = min(prior * Cfn, (1 - prior) * Cfp)
    return DCFu / B_dummy


def binary_min_detection_cost(ll_ratio, L, prior, Cfn, Cfp):
    thresholds = np.array(ll_ratio)
    thresholds.sort()
    thresholds = np.concatenate(
        [np.array([-np.inf]), thresholds, np.array([np.inf])])
    thresholds_size = thresholds.shape[0]
    DCFs = np.zeros(thresholds_size)
    for i, t in enumerate(thresholds):
        DCFs[i] = binary_evaluation(ll_ratio, L, prior, Cfn, Cfp, t)
    return np.min(DCFs)


def binary_plot_roc(ll_ratio, L):
    thresholds = np.array(ll_ratio)
    thresholds.sort()
    thresholds = np.concatenate(
        [np.array([-np.inf]), thresholds, np.array([np.inf])])
    thresholds_size = thresholds.shape[0]
    FPR_vector = np.zeros(thresholds_size)
    FNR_vector = np.zeros(thresholds_size)
    for i, t in enumerate(thresholds):
        _, FPR, FNR, _ = prml.optimal_bayes(ll_ratio, L, 0, 1, 1, t)
        FPR_vector[i] = FPR
        FNR_vector[i] = FNR
    plt.plot(FPR_vector, 1 - FNR_vector)
    plt.show()


def binary_plot_error(ll_ratio, L, colors=('r', 'b'), Cfn=1, Cfp=1):
    effPriorLogOdds = np.linspace(-3, 3, 21)
    effPriorLogOddsSize = effPriorLogOdds.shape[0]
    effectivePrior = 1 / (1 + np.exp(-effPriorLogOdds))
    DCF = np.zeros(effPriorLogOddsSize)
    minDCF = np.zeros(effPriorLogOddsSize)
    for i, prior in enumerate(effectivePrior):
        DCF[i] = binary_evaluation(ll_ratio, L, prior, Cfn, Cfp)
        minDCF[i] = binary_min_detection_cost(ll_ratio, L, prior,
                                              Cfn, Cfp)
    plt.plot(effPriorLogOdds, DCF, label='DCF', color=colors[0])
    plt.plot(effPriorLogOdds, minDCF, label='minDCF', color=colors[1])
    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
