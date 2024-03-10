import torch.nn as nn
import torch.nn.functional as F
from MLEC.models.BertEncoder import BertEncoder
from MLEC.models.MLECModel import MLECModel
from transformers import (
    EncoderDecoderModel,
    BertConfig,
    EncoderDecoderConfig,
)
import torch
import numpy as np


class SpanEmoB2B(MLECModel):

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
        super(SpanEmoB2B, self).__init__(
            alpha=alpha,
            beta=beta,
        )
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        decoder_config = BertConfig.from_pretrained(
            "bert-base-uncased", output_hidden_states=True
        )
        self.config = EncoderDecoderConfig.from_encoder_decoder_configs(
            encoder_config, decoder_config
        )
        self.model = EncoderDecoderModel(self.config)
        self.model.encoder.resize_token_embeddings(embedding_vocab_size)
        self.model.decoder.resize_token_embeddings(embedding_vocab_size)
        self.ffn = nn.Sequential(
            nn.Linear(decoder_config.hidden_size, label_size),
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
        all_label_input_ids = all_label_input_ids.long().to(device)
        attention_masks = attention_masks.to(device)
        label_attention_masks = label_attention_masks.to(device)
        inputs, num_rows = inputs.long().to(device), inputs.size(0)
        targets = targets.float().to(device)
        label_idxs, label_input_ids = label_idxs[0].long().to(
            device
        ), label_input_ids.long().to(device)
        outputs = self.model(
            inputs,
            attention_mask=attention_masks,
            decoder_input_ids=label_input_ids,
            decoder_attention_mask=label_attention_masks,
        )
        # get logits
        # print(outputs.decoder_hidden_states)
        hidden_states = (
            outputs.decoder_hidden_states[-1]
            .clone()
            .detach()
            .requires_grad_(True)
            .to(device)
        )
        logits = self.ffn(torch.tensor(outputs.decoder_hidden_states[-1]).to(device))
        batch_size, sequence_lengths, _ = outputs.logits.shape
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if inputs is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = (
                    torch.eq(inputs, self.config.pad_token_id).int().argmax(-1) - 1
                )
                sequence_lengths = sequence_lengths % inputs.shape[-1]
                sequence_lengths = sequence_lengths.to(device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[
            torch.arange(batch_size, device=device), sequence_lengths
        ]

        # get probabilities of tokens
        # get the predictions
        y_pred = self.compute_pred(pooled_logits)

        return num_rows, y_pred, pooled_logits, targets, outputs
