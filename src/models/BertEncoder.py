from transformers import BertModel, AutoModel
import torch.nn.functional as F
import torch.nn as nn
import torch
import transformers


class BertEncoder(nn.Module):
    def __init__(self, lang="English"):
        """
        :param lang: str, train bert encoder for a given language
        """
        super(BertEncoder, self).__init__()
        if lang == "English":
            self.bert = BertModel.from_pretrained("bert-base-uncased")
        elif lang == "Arabic":
            self.bert = AutoModel.from_pretrained("asafaya/bert-base-arabic")
        elif lang == "Spanish":
            self.bert = AutoModel.from_pretrained(
                "dccuchile/bert-base-spanish-wwm-uncased"
            )
        self.feature_size = self.bert.config.hidden_size

    def forward(self, input_ids):
        """
        :param input_ids: list[str], list of tokenised sentences
        :return: last hidden representation, torch.tensor of shape (batch_size, seq_length, hidden_dim)
        """
        if int((transformers.__version__)[0]) == 4:
            last_hidden_state = self.bert(input_ids=input_ids).last_hidden_state
        else:  # transformers version should be as indicated in the requirements.txt file
            last_hidden_state, pooler_output = self.bert(input_ids=input_ids)
        return last_hidden_state
