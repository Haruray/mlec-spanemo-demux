# base class for multi-label emotion classification
import torch.nn.functional as F
import torch
from MLEC.emotion_corr_weightings.Correlations import Correlations


def intra_corr_loss(y_hat, y_true, correlations: Correlations, reduction="mean"):
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
                    correlations.get(
                        index=(
                            (
                                list(correlations.col_names[y_o.squeeze().tolist()])
                                if type(y_o.squeeze().tolist()) is not int
                                else [correlations.col_names[y_o.squeeze().tolist()]]
                            ),
                            (
                                list(correlations.col_names[y_o.squeeze().tolist()])
                                if type(y_o.squeeze().tolist()) is not int
                                else [correlations.col_names[y_o.squeeze().tolist()]]
                            ),
                        ),
                        decreasing=False,
                    ).to(device)
                )
                .squeeze(-1)
            ).sum()
            # calculate the number of comparisons y_o.size(0) ^2 - y_o.size(0)
            num_comparisons = y_o.size(0) ** 2 - y_o.size(0)
            if num_comparisons == 0:
                num_comparisons = 1
            # print(num_comparisons)
            loss_present[idx] = output.div(num_comparisons)

        if y_z.nelement() > 0:
            # calculate the exp of difference between each pair of y_z, then sum them all
            output = torch.exp(
                torch.add(y_h[y_z], y_h[y_z][:, None])
                .mul(
                    correlations.get(
                        index=(
                            (
                                list(correlations.col_names[y_z.squeeze().tolist()])
                                if type(y_z.squeeze().tolist()) is not int
                                else [correlations.col_names[y_z.squeeze().tolist()]]
                            ),
                            (
                                list(correlations.col_names[y_z.squeeze().tolist()])
                                if type(y_z.squeeze().tolist()) is not int
                                else [correlations.col_names[y_z.squeeze().tolist()]]
                            ),
                        ),
                        decreasing=False,
                    ).to(device)
                )
                .squeeze(-1)
            ).sum()
            num_comparisons = y_z.size(0) ** 2 - y_z.size(0)
            loss_absent[idx] = output.div(num_comparisons)
    total_loss = (loss_present.mean() + loss_absent.mean()) / 2
    return total_loss
