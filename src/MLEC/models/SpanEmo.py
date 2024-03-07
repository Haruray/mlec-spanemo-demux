import torch.nn.functional as F
import torch.nn as nn
from MLEC.models.BertEncoder import BertEncoder
from MLEC.emotion_corr_weightings.Correlations import Correlations
from MLEC.enums.CorrelationType import CorrelationType
from MLEC.models.MLECModel import MLECModel


class SpanEmo(MLECModel):
    def __init__(
        self,
        output_dropout=0.1,
        lang="English",
        alpha=0.2,
        beta=0.1,
        joint_loss=False,
        corr_type=CorrelationType.IDENTITY,
        col_names=[],
    ):
        """casting multi-label emotion classification as span-extraction
        :param output_dropout: The dropout probability for output layer
        :param lang: encoder language
        :param joint_loss: which loss to use cel|corr|cel+corr
        :param alpha: control contribution of each loss function in case of joint training
        """
        super(SpanEmo, self).__init__(corr_type=corr_type, col_names=col_names)
        self.bert = BertEncoder(lang=lang)
        self.alpha = alpha
        self.beta = beta
        self.joint_loss = joint_loss
        self.correlations = Correlations(
            corr_type=corr_type, col_names=col_names, active=True
        )

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

        # Loss Function
        loss_binary_ce = F.binary_cross_entropy_with_logits(logits, targets).to(device)
        loss_inter_corr = self.inter_corr_loss(logits, targets)
        loss_intra_corr = self.intra_corr_loss(logits, targets)
        loss_corr_joint = (
            0.5 * (loss_inter_corr + loss_intra_corr) if self.joint_loss else 0
        )
        loss = (
            (1 - self.alpha - self.beta) * loss_binary_ce
            + self.alpha * loss_corr_joint
            + self.beta * loss_inter_corr
        )

        y_pred = self.compute_pred(logits)
        return loss, num_rows, y_pred, targets.cpu().numpy()
