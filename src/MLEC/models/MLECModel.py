# base class for multi-label emotion classification
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from MLEC.emotion_corr_weightings.Correlations import Correlations
from MLEC.enums.CorrelationType import CorrelationType


class MLECModel(nn.Module):

    def __init__(
        self,
        alpha=0.2,
        beta=0.1,
    ):
        """casting multi-label emotion classification as span-extraction
        :param output_dropout: The dropout probability for output layer
        :param lang: encoder language
        :param joint_loss: which loss to use cel|corr|cel+corr
        :param alpha: control contribution of each loss function in case of joint training
        """
        super(MLECModel, self).__init__()

    def forward(self, batch, device):
        """
        :param batch: tuple of (input_ids, labels, length, label_indices)
        :param device: device to run calculations on
        :return: loss, num_rows, y_pred, targets
        """
        pass

    def compute_pred(self, logits, threshold=0.5):
        """
        :param logits: model predictions
        :param threshold: threshold value
        :return:
        """
        y_pred = torch.sigmoid(logits) > threshold
        return y_pred.float().cpu().numpy()
