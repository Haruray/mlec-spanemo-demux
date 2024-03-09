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
import numpy as np


class SpanEmoB2B(MLECDecoder):

    def __init__(
        self,
        output_dropout=0.1,
        embedding_vocab_size=30522,
        alpha=0.2,
        beta=0.1,
        label_size=11,
        batch_size=32,
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
                batch_size,
                label_size,
            ),
            nn.Tanh(),
            nn.Dropout(p=output_dropout),
            nn.Linear(label_size, 1),
        )
        self.encoder_parameters = self.model.encoder.parameters()
        self.decoder_parameters = self.model.decoder.parameters()

    def forward(self, batch, device):
        """
        :param batch: tuple of (input_ids, labels, length, label_indices)
        :param device: device to run calculations on
        :return: loss, num_rows, y_pred, targets
        """
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
        label_attention_masks = label_attention_masks.to(device)
        inputs, num_rows = inputs.long().to(device), inputs.size(0)
        label_idxs, label_input_ids = label_idxs[0].long().to(
            device
        ), label_input_ids.long().to(device)
        outputs = self.model(
            inputs,
            attention_mask=attention_masks,
            decoder_input_ids=label_input_ids,
            decoder_attention_mask=label_attention_masks,
        )
        outputs_logits = outputs.logits[0][-1].cpu().detach().numpy()
        # get the logits of the all_label_input_ids
        emotion_logits = []
        for label_input_id in all_label_input_ids:
            emotion_logits.append(outputs_logits[label_input_id])
        # print(outputs)
        outputs_logits = torch.tensor(np.array(emotion_logits)).to(device)
        logits = (
            self.ffn(outputs_logits).squeeze(-1).index_select(dim=1, index=label_idxs)
        )
        print(logits.shape)
        print(logits)
        y_pred = self.compute_pred(logits.to(device))
        logits = outputs[0]

        return num_rows, y_pred, logits, targets, outputs
