import torch.nn.functional as F
import torch.nn as nn
from MLEC.models.BertEncoder import BertEncoder
from MLEC.emotion_corr_weightings.Correlations import Correlations
from MLEC.enums.CorrelationType import CorrelationType
from MLEC.models.MLECModel import MLECEncoder


class SpanEmo(MLECEncoder):
    def __init__(
        self,
        output_dropout=0.1,
        lang="English",
        alpha=0.2,
        beta=0.1,
    ):
        """casting multi-label emotion classification as span-extraction
        :param output_dropout: The dropout probability for output layer
        :param lang: encoder language
        :param joint_loss: which loss to use cel|corr|cel+corr
        :param alpha: control contribution of each loss function in case of joint training
        """
        super(SpanEmo, self).__init__()
        self.bert = BertEncoder(lang=lang)
        self.alpha = alpha
        self.beta = beta

        self.ffn = nn.Sequential(
            nn.Linear(self.bert.feature_size, self.bert.feature_size),
            nn.Tanh(),
            nn.Dropout(p=output_dropout),
            nn.Linear(self.bert.feature_size, 1),
        )

    def forward(self, batch, device):
        """
        :param batch: tuple of (input_ids, labels, length, label_indices)
        :param device: device to run calculations on
        :return: loss, num_rows, y_pred, targets
        """
        # prepare inputs and targets
        inputs, targets, lengths, label_idxs = batch
        inputs, num_rows = inputs.to(device), inputs.size(0)
        label_idxs, targets = label_idxs[0].long().to(device), targets.float().to(
            device
        )

        # Bert encoder
        last_hidden_state = self.bert(inputs)

        # FFN---> 2 linear layers---> linear layer + tanh---> linear layer
        # select span of labels to compare them with ground truth ones
        logits = (
            self.ffn(last_hidden_state)
            .squeeze(-1)
            .index_select(dim=1, index=label_idxs)
        )

        y_pred = self.compute_pred(logits)
        return num_rows, y_pred, logits, targets
