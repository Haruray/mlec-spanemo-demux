from transformers import BertModel, AutoModel
import torch.nn.functional as F
import torch.nn as nn
import transformers


class BertEncoder(nn.Module):
    def __init__(self, lang="English"):
        """
        :param lang: str, train bert encoder for a given language
        """
        super(BertEncoder, self).__init__()
        if lang == "English":
            self.bert = BertModel.from_pretrained("bert-base-uncased")
        elif lang == "Indonesia":
            self.bert = AutoModel.from_pretrained("indolem/indobert-base-uncased")
        self.feature_size = self.bert.config.hidden_size

    def forward(self, input_ids, attention_mask=None):
        """
        :param input_ids: list[str], list of tokenised sentences
        :return: last hidden representation, torch.tensor of shape (batch_size, seq_length, hidden_dim)
        """
        last_hidden_state, _ = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=False
        )
        return last_hidden_state
