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
        output_dropout=0.1,
        lang="English",
        alpha=0.2,
        beta=0.1,
        joint_loss=True,
        corr_type=CorrelationType.IDENTITY,
        col_names=[],
    ):
        """casting multi-label emotion classification as span-extraction
        :param output_dropout: The dropout probability for output layer
        :param lang: encoder language
        :param joint_loss: which loss to use cel|corr|cel+corr
        :param alpha: control contribution of each loss function in case of joint training
        """
        super(MLECModel, self).__init__()
        self.correlations = Correlations(
            corr_type=corr_type, col_names=col_names, active=True
        )
        self.col_names = np.array(col_names)

    def forward(self, batch, device):
        """
        :param batch: tuple of (input_ids, labels, length, label_indices)
        :param device: device to run calculations on
        :return: loss, num_rows, y_pred, targets
        """
        pass

    def inter_corr_loss(self, y_hat, y_true, reduction="mean"):
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
                output = torch.exp(
                    torch.sub(y_h[y_z], y_h[y_o][:, None])
                    .mul(
                        self.correlations.get(
                            index=(
                                (
                                    (list(self.col_names[y_o.squeeze().tolist()]))
                                    if type(y_o.squeeze().tolist()) is not int
                                    else [self.col_names[y_o.squeeze().tolist()]]
                                ),
                                (
                                    list(self.col_names[y_z.squeeze().tolist()])
                                    if type(y_z.squeeze().tolist()) is not int
                                    else [self.col_names[y_z.squeeze().tolist()]]
                                ),
                            ),
                            decreasing=True,
                        )
                    )
                    .squeeze(-1)
                ).sum()
                num_comparisons = y_z.size(0) * y_o.size(0)
                # multiply the output by the correlation matrix
                # self.correlations.get() and the arguments are the emotions that are being compared
                # get the names from self.col_names
                loss[idx] = output.div(num_comparisons)
        return loss.mean() if reduction == "mean" else loss.sum()

    def intra_corr_loss(self, y_hat, y_true, reduction="mean"):
        """
        :param y_hat: model predictions, shape(batch, classes)
        :param y_true: target labels (batch, classes)
        :param reduction: whether to avg or sum loss
        :return: loss
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        loss_present = torch.zeros(y_true.size(0)).to(device)
        loss_absent = torch.zeros(y_true.size(0)).to(device)
        for idx, (y, y_h) in enumerate(zip(y_true, y_hat.sigmoid())):
            # y_z: indices of zeros in y, y_o: indices of ones in y
            y_z, y_o = (y == 0).nonzero(), y.nonzero()
            if y_o.nelement() > 0:
                # if y_o has more than one element, then calculate loss
                # calculate the exp of difference between each pair of negative y_o, then sum them all
                # formula goes something like this exp(-(y_h[i] - -y_h[j])) for all i,j in y_o
                output = torch.exp(
                    torch.add(-y_h[y_o], -y_h[y_o][:, None])
                    .mul(
                        self.correlations.get(
                            index=(
                                (
                                    list(self.col_names[y_o.squeeze().tolist()])
                                    if type(y_o.squeeze().tolist()) is not int
                                    else [self.col_names[y_o.squeeze().tolist()]]
                                ),
                                (
                                    list(self.col_names[y_o.squeeze().tolist()])
                                    if type(y_o.squeeze().tolist()) is not int
                                    else [self.col_names[y_o.squeeze().tolist()]]
                                ),
                            ),
                            decreasing=False,
                        )
                    )
                    .squeeze(-1)
                ).sum()
                # calculate the number of comparisons y_o.size(0) ^2 - y_o.size(0)
                num_comparisons = y_o.size(0) ** 2 - y_o.size(0)
                loss_present[idx] = output.div(num_comparisons)

            if y_z.nelement() > 0:
                # calculate the exp of difference between each pair of y_z, then sum them all
                output = torch.exp(
                    torch.add(y_h[y_z], y_h[y_z][:, None])
                    .mul(
                        self.correlations.get(
                            index=(
                                (
                                    list(self.col_names[y_z.squeeze().tolist()])
                                    if type(y_z.squeeze().tolist()) is not int
                                    else [self.col_names[y_z.squeeze().tolist()]]
                                ),
                                (
                                    list(self.col_names[y_z.squeeze().tolist()])
                                    if type(y_z.squeeze().tolist()) is not int
                                    else [self.col_names[y_z.squeeze().tolist()]]
                                ),
                            ),
                            decreasing=False,
                        )
                    )
                    .squeeze(-1)
                ).sum()
                num_comparisons = y_z.size(0) ** 2 - y_z.size(0)
                loss_absent[idx] = output.div(num_comparisons)
        total_loss = (loss_present.mean() + loss_absent.mean()) / 2
        return total_loss

    def compute_pred(self, logits, threshold=0.5):
        """
        :param logits: model predictions
        :param threshold: threshold value
        :return:
        """
        y_pred = torch.sigmoid(logits) > threshold
        return y_pred.float().cpu().numpy()
