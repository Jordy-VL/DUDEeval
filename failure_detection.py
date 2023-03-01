from __future__ import annotations
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn import metrics as skm

AURC_DISPLAY_SCALE = 1 #1000

"""
From: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1204/reports/custom/report52.pdf 

The risk-coverage (RC) curve [28, 16] is a measure of the trade-off between the
coverage (the proportion of test data encountered), and the risk (the error rate under this coverage). Since each
prediction comes with a confidence score, given a list of prediction correctness Z paired up with the confidence
scores C, we sort C in reverse order to obtain sorted C'
, and its corresponding correctness Z'
. Note that the correctness is computed based on Exact Match (EM) as described in [22]. The RC curve is then obtained by
computing the risk of the coverage from the beginning of Z'
(most confident) to the end (least confident). In particular, these metrics evaluate 
the relative order of the confidence score, which means that we want wrong
answers have lower confidence score than the correct ones, ignoring their absolute values. 

Source: https://github.com/kjdhfg/fd-shifts 

References:
-----------

[1] Jaeger, P.F., Lüth, C.T., Klein, L. and Bungert, T.J., 2022. A Call to Reflect on Evaluation Practices for Failure Detection in Image Classification. arXiv preprint arXiv:2211.15259.

[2] Kamath, A., Jia, R. and Liang, P., 2020. Selective Question Answering under Domain Shift. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (pp. 5684-5696).

"""

@dataclass
class StatsCache:
    """Cache for stats computed by scikit used by multiple metrics.

    Attributes:
        confids (array_like): Confidence values
        correct (array_like): Boolean array (best converted to int) where predictions were correct
    """

    confids: npt.NDArray[Any]
    correct: npt.NDArray[Any]

    @cached_property
    def roc_curve_stats(self) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        fpr, tpr, _ = skm.roc_curve(self.correct, self.confids)
        return fpr, tpr

    @property
    def residuals(self) -> npt.NDArray[Any]:
        return 1 - self.correct

    @cached_property
    def rc_curve_stats(self) -> tuple[list[float], list[float], list[float]]:
        coverages = []
        risks = []

        n_residuals = len(self.residuals)
        idx_sorted = np.argsort(self.confids)

        coverage = n_residuals
        error_sum = sum(self.residuals[idx_sorted])

        coverages.append(coverage / n_residuals)
        risks.append(error_sum / n_residuals)

        weights = []

        tmp_weight = 0
        for i in range(0, len(idx_sorted) - 1):
            coverage = coverage - 1
            error_sum = error_sum - self.residuals[idx_sorted[i]]
            selective_risk = error_sum / (n_residuals - 1 - i)
            tmp_weight += 1
            if i == 0 or self.confids[idx_sorted[i]] != self.confids[idx_sorted[i - 1]]:
                coverages.append(coverage / n_residuals)
                risks.append(selective_risk)
                weights.append(tmp_weight / n_residuals)
                tmp_weight = 0

        # add a well-defined final point to the RC-curve.
        if tmp_weight > 0:
            coverages.append(0)
            risks.append(risks[-1])
            weights.append(tmp_weight / n_residuals)
        return coverages, risks, weights

def AUROC_PR(pred_known, pred_unknown):
    neg = list(np.max(pred_known, axis=-1))
    pos = list(np.max(pred_unknown, axis=-1))
    auroc, aupr = compute_auc_aupr(neg, pos, pos_label=0)
    return auroc, aupr


def compute_auc_aupr(neg, pos, pos_label=1): #zeros are known; ones are unknown
    ys = np.concatenate((np.zeros(len(neg)), np.ones(len(pos))), axis=0)
    neg = np.array(neg)[np.logical_not(np.isnan(neg))]
    pos = np.array(pos)[np.logical_not(np.isnan(pos))]
    scores = np.concatenate((neg, pos), axis=0)
    auc = skm.roc_auc_score(ys, scores)  # AUROC ##1 as default
    aupr = skm.average_precision_score(ys, scores)  # AUPR
    if pos_label == 1:
        return auc, aupr
    else:
        return 1 - auc, 1 - aupr

def failauc(stats_cache: StatsCache) -> float:
    """AUROC_f metric function
    Args:
        stats_cache (StatsCache): StatsCache object
    Returns:
        metric value
    """
    fpr, tpr = stats_cache.roc_curve_stats
    return skm.auc(fpr, tpr)

def aurc(stats_cache: StatsCache):
    """auc metric function
    Args:
        stats_cache (StatsCache): StatsCache object
    Returns:
        metric value
    Important for assessment: LOWER is better!
    """
    _, risks, weights = stats_cache.rc_curve_stats
    return (
        sum([(risks[i] + risks[i + 1]) * 0.5 * weights[i] for i in range(len(weights))])
        * AURC_DISPLAY_SCALE
    )

def aurc_phase2(references, predictions):
    cache = StatsCache(confids=predictions, correct=references)
    return {'aurc': aurc(cache)}

def AUROC_phase2(references, predictions):
    cache = StatsCache(confids=predictions, correct=references)
    return {'AUROC': failauc(cache)}


def test_aurc():
    """
    Three cases from https://openreview.net/pdf?id=YnkGMIh0gvX

    separable_less_accurate_references ; acc 2/5, AUROC 1
    unseparable_lowcorrect_references ; acc 3/5, AUROC 0.75
    unseparable_highincorrect_references ; acc 3/5, AUROC 0.583
    """
    predictions = np.array([0.9, 0.1, 0.3, 1.0, 0.1])
    separable_less_accurate_references = np.array([1, 0, 0, 1, 0])
    result = aurc_phase2(separable_less_accurate_references, predictions)
    print(f"separable_less_accurate gives an AURC of {result}")

    unseparable_lowcorrect_references = np.array([1, 1, 0, 1, 0])
    result = aurc_phase2(unseparable_lowcorrect_references, predictions)
    print(f"unseparable_lowcorrect gives an AURC of {result}") # BEST!

    unseparable_highincorrect_references = np.array([0, 1, 1, 1, 0])
    result = aurc_phase2(unseparable_highincorrect_references, predictions)
    print(f"unseparable_highincorrect gives an AURC of {result}")

def test_ood():
    """
    Simple example following methodology in https://ieeexplore.ieee.org/document/9761166
    
    * Reversed labeling of IID vs OOD, just use pos_label=1 
    """
    gt = [1, 0, 1, 0, 1, 1, 1, 1, 0] #1 being IID, 0 being IID
    predictions = [0.6648081 , 0.98290163, 0.79909354, 0.9961113 , 0.1472904 ,
       0.29210454, 0.0049987 , 0.70650965, 0.97676945]
    result = AUROC_phase2(gt, predictions)
    print(f"worst AUROC_ood of {result}")
    '''
    # p_iid = [0.66, 0.8, 0.14, 0.29, 0.004, 0.7]
    # p_ood = [0.98, 0.99, 0.97] 
    #indeed AUROC is 0 -> ood ranked higher than iid
    '''
    gt = [0, 1, 0, 1, 1, 1, 1, 1, 0] #
    result = AUROC_phase2(gt, predictions)
    print(f"mixed AUROC_ood of {result}")
    
    gt = np.logical_not([1, 0, 1, 0, 1, 1, 1, 1, 0]) #1 being IID
    result = AUROC_phase2(gt, predictions)
    print(f"perfect AUROC_ood of {result}")
    
if __name__ == "__main__":
    test_aurc()
    test_ood()