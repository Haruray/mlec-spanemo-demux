import torch.nn as nn
import torch.nn.functional as F
from MLEC.models.BertEncoder import BertEncoder
from MLEC.models.MLECEncoder import MLECEncoder
from MLEC.models.MLECDecoder import MLECDecoder
from transformers import (
    EncoderDecoderModel,
    BertConfig,
    EncoderDecoderConfig,
)
import torch


class SpanEmoB2B(MLECDecoder):

    def __init__(
        self,
        output_dropout=0.1,
        embedding_vocab_size=30522,
        alpha=0.2,
        beta=0.1,
    ):
        """casting multi-label emotion classification as span-extraction
        :param output_dropout: The dropout probability for output layer
        :param lang: encoder language
        :param joint_loss: which loss to use cel|corr|cel+corr
        :param alpha: control contribution of each loss function in case of joint training
        """
        super(SpanEmoB2B, self).__init__()
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        decoder_config = BertConfig.from_pretrained("bert-base-uncased")
        config = EncoderDecoderConfig.from_encoder_decoder_configs(
            encoder_config, decoder_config
        )
        self.model = EncoderDecoderModel(config)
        self.model.encoder.resize_token_embeddings(embedding_vocab_size)
        self.model.decoder.resize_token_embeddings(embedding_vocab_size)
        self.ffn = nn.Sequential(
            nn.Linear(
                self.model.decoder.config.hidden_size,
                self.model.decoder.config.hidden_size,
            ),
            nn.Tanh(),
            nn.Dropout(p=output_dropout),
            nn.Linear(self.model.decoder.config.hidden_size, 1),
        )
        self.encoder_parameters = self.model.encoder.parameters()
        self.decoder_parameters = self.model.decoder.parameters()

    def forward(self, batch, device):
        """
        :param batch: tuple of (input_ids, labels, length, label_indices)
        :param device: device to run calculations on
        :return: loss, num_rows, y_pred, targets
        """
        inputs, targets, lengths, label_idxs = batch
        inputs, num_rows = inputs.to(device).to(torch.int64), inputs.size(0)
        label_idxs, targets = label_idxs[0].long().to(device), targets.float().to(
            device
        )
        outputs = self.model(
            inputs,
            decoder_input_ids=targets,
        )
        outputs_logits = outputs[0]
        logits = (
            self.ffn(outputs_logits).squeeze(-1).index_select(dim=1, index=label_idxs)
        )

        y_pred = self.compute_pred(logits)
        logits = outputs[0]

        return num_rows, y_pred, logits, targets, outputs
