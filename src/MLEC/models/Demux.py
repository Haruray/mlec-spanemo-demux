import torch.nn.functional as F
import torch.nn as nn
import torch
from MLEC.models.BertEncoder import BertEncoder
from MLEC.enums.CorrelationType import CorrelationType
from MLEC.models.MLECModel import MLECModel


class Demux(MLECModel):

    def __init__(
        self,
        output_dropout=0.1,
        lang="English",
        alpha=0.2,
        beta=0.1,
        embedding_vocab_size=30522,
        label_size=11,
        output_size=1,
        device="cuda:0",
    ):
        """casting multi-label emotion classification as span-extraction
        :param output_dropout: The dropout probability for output layer
        :param lang: encoder language
        :param joint_loss: which loss to use cel|corr|cel+corr
        :param alpha: control contribution of each loss function in case of joint training
        """
        super(Demux, self).__init__(
            alpha=alpha,
            beta=beta,
        )
        self.encoder = BertEncoder(lang=lang)
        self.encoder.bert.resize_token_embeddings(embedding_vocab_size)
        self.encoder.bert.to(device)

        self.ffn = nn.Sequential(
            nn.Linear(self.encoder.feature_size, self.encoder.feature_size),
            nn.Tanh(),
            nn.Dropout(p=output_dropout),
            nn.Linear(self.encoder.feature_size, 1),
        ).to(device)
        self.encoder_parameters = self.encoder.parameters()

    def forward(
        self,
        input_ids,
        input_attention_masks,
        targets=None,
        target_input_ids=None,
        target_attention_masks=None,
        device="cuda:0",
        **kwargs
    ):
        """
        :param batch: tuple of (input_ids, labels, length, label_indices)
        :param device: device to run calculations on
        :return: loss, num_rows, y_pred, targets
        """
        # prepare inputs and targets
        lengths = kwargs.get("lengths", None)
        label_idxs = kwargs.get("label_idxs", None)

        input_attention_masks = input_attention_masks.to(device)
        input_ids, num_rows = input_ids.to(device), input_ids.size(0)

        if label_idxs is not None:
            label_idxs = label_idxs.long().to(device)

        if targets is not None:
            targets = targets.float().to(device)

        # Bert encoder
        last_hidden_state = self.encoder(
            input_ids, attention_mask=input_attention_masks
        )

        # take only the emotion embeddings
        last_emotion_state = [
            torch.stack(
                [
                    last_hidden_state.index_select(dim=1, index=inds).mean(1)
                    for inds in emo_inds
                ],
                dim=1,
            )
            for emo_inds in label_idxs
        ]

        # FFN---> 2 linear layers---> linear layer + tanh---> linear layer
        # select span of labels to compare them with ground truth ones
        logits = torch.stack(
            [self.ffn(cluster_stack).max(1)[0] for cluster_stack in last_emotion_state],
            dim=1,
        )
        print(logits.size())
        logits = logits.squeeze(-1)
        print(logits.size())
        print(logits[0])
        y_pred = self.compute_pred(logits)
        return num_rows, y_pred, logits, targets, last_hidden_state
