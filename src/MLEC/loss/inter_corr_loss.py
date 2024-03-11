# base class for multi-label emotion classification
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from MLEC.emotion_corr_weightings.Correlations import Correlations
from MLEC.enums.CorrelationType import CorrelationType


def inter_corr_loss(y_hat, y_true, correlations: Correlations, reduction="mean"):
    """
    :param y_hat: model predictions, shape(batch, classes)
    :param y_true: target labels (batch, classes)
    :param reduction: whether to avg or sum loss
    :return: loss
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss = torch.zeros(y_true.size(0)).to(device)
    for idx, (y, y_h) in enumerate(zip(y_true, y_hat.sigmoid())):
        # y_z: indices of zeros in y, y_o: indices of ones in y
        y_z, y_o = (y == 0).nonzero(), y.nonzero()
        # if there are no ones in y, then loss is zero
        # no ones in y means no positive examples
        if y_o.nelement() > 0:
            output = torch.exp(torch.sub(y_h[y_z], y_h[y_o][:, None]).squeeze(-1)).sum()
            num_comparisons = y_z.size(0) * y_o.size(0)
            # multiply the output by the correlation matrix
            # self.correlations.get() and the arguments are the emotions that are being compared
            # get the names from self.col_names
            loss[idx] = output.div(num_comparisons)
    return loss.mean() if reduction == "mean" else loss.sum()
