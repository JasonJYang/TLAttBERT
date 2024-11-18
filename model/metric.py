import torch
import numpy as np
from sklearn import metrics
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score

def rmse(y_pred, y_true):
    return metrics.mean_squared_error(y_pred=y_pred, y_true=y_true, squared=False)

def pearson_roi(y_pred, y_true):
    return pearsonr(x=y_pred, y=y_true)[0]

def pearson_pval(y_pred, y_true):
    return pearsonr(x=y_pred, y=y_true)[1]

def r_squared(y_pred, y_true):
    return metrics.r2_score(y_pred=y_pred, y_true=y_true)
