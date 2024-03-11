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

        self.ffn = nn.Sequential(
            nn.Linear(label_size, label_size),
            nn.Tanh(),
            nn.Dropout(p=output_dropout),
            nn.Linear(label_size, 1),
        )
        self.encoder_parameters = self.encoder.parameters()

    def forward(self, batch, device):
        """
        :param batch: tuple of (input_ids, labels, length, label_indices)
        :param device: device to run calculations on
        :return: loss, num_rows, y_pred, targets
        """
        # prepare inputs and targets
        (
            inputs,
            attention_masks,
            targets,
            lengths,
            label_idxs,
            label_input_ids,
            label_attention_masks,
            all_label_input_ids,
        ) = batch
        attention_masks = attention_masks.to(device)
        inputs, num_rows = inputs.to(device), inputs.size(0)
        label_idxs, targets = label_idxs[0].long().to(device), targets.float().to(
            device
        )

        # Bert encoder
        last_hidden_state = self.encoder(inputs)

        # the embedding clipping, take only the emotions embedding
        last_emotion_state = [
            torch.stack(
                [
                    last_hidden_state.index_select(
                        dim=1, index=inds.to(last_hidden_state.device)
                    ).mean(1)
                    for inds in emo_inds
                ],
                dim=1,
            )
            for emo_inds in label_idxs
        ]

        # FFN---> 2 linear layers---> linear layer + tanh---> linear layer
        # select span of labels to compare them with ground truth ones
        logits = self.ffn(last_emotion_state).squeeze(-1)
        y_pred = self.compute_pred(logits)
        return num_rows, y_pred, logits, targets
