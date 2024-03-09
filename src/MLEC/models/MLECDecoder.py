# base class for multi-label emotion classification
import torch.nn.functional as F
import torch.nn as nn
import torch
from MLEC.layers.PositionalEncoding import PositionalEncoding


class MLECDecoder(nn.Module):

    def __init__(
        self,
    ):
        super(MLECDecoder, self).__init__()

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
