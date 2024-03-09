import torch.nn as nn
import torch.nn.functional as F
from MLEC.models.BertEncoder import BertEncoder
from MLEC.models.MLECEncoder import MLECEncoder
from MLEC.models.MLECDecoder import MLECDecoder


class SpanEmoEncoderDecoder(MLECDecoder):

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        num_layers,
        lang="English",
        output_dropout=0.1,
        embedding_vocab_size=30522,
        output_size=1,
    ):
        """casting multi-label emotion classification as span-extraction
        :param output_dropout: The dropout probability for output layer
        :param lang: encoder language
        :param joint_loss: which loss to use cel|corr|cel+corr
        :param alpha: control contribution of each loss function in case of joint training
        """
        super(SpanEmoEncoderDecoder, self).__init__(
            d_model,
            nhead,
            dim_feedforward,
            num_layers,
            output_dropout,
            embedding_vocab_size,
            lang,
        )
        self.output_dropout = output_dropout
        self.encoder = BertEncoder(lang=lang)
        self.encoder.bert.resize_token_embeddings(embedding_vocab_size)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, output_dropout
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        # Final layer normalization
        self.layer_norm = nn.LayerNorm(self.encoder.feature_size)
        self.ffn = nn.Sequential(
            nn.Linear(self.encoder.feature_size, self.encoder.feature_size),
            nn.Tanh(),
            nn.Dropout(p=output_dropout),
            nn.Linear(output_size, 1),
        )

    def forward(
        self,
        target_sequence,
        encoder_output_embedding,
        target_mask=None,
        encoder_output_mask=None,
        target_key_padding_mask=None,
        encoder_output_key_padding_mask=None,
    ):
        """
        :param batch: tuple of (input_ids, labels, length, label_indices)
        :param device: device to run calculations on
        :return: loss, num_rows, y_pred, targets
        """
        target_embedding = self.encoder(target_sequence)
        # apply dropout
        target_embedding = F.dropout(
            target_embedding, p=self.output_dropout, training=self.training
        )
        output = self.decoder(
            target_embedding,
            encoder_output_embedding,
            tgt_mask=target_mask,
            memory_mask=encoder_output_mask,
            tgt_key_padding_mask=target_key_padding_mask,
            memory_key_padding_mask=encoder_output_key_padding_mask,
        )
        output = self.layer_norm(output)
        # classification
        # FFN---> 2 linear layers---> linear layer + tanh---> linear layer
        # select span of labels to compare them with ground truth ones
        logits = self.ffn(output).squeeze(-1)

        y_pred = self.compute_pred(logits)
        return y_pred, logits
