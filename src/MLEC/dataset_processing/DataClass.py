from torch.utils.data import Dataset
from transformers import BertTokenizer, AutoTokenizer
from tqdm import tqdm
import torch
import pandas as pd
from MLEC.dataset_processing.twitter_preprocessor import twitter_preprocessor


class DataClass(Dataset):
    def __init__(self, args, filename):
        self.args = args
        self.filename = filename
        self.max_length = int(args["--max-length"])
        self.data, self.labels, self.label_names = self.load_dataset()
        self.all_label_input_ids = []

        if args["--lang"] == "English":
            self.bert_tokeniser = BertTokenizer.from_pretrained(
                "bert-base-uncased", do_lower_case=True
            )
            vocab = self.bert_tokeniser.get_vocab()
            self.bert_tokeniser.add_tokens(["pessimism"])
        elif args["--lang"] == "Arabic":
            self.bert_tokeniser = AutoTokenizer.from_pretrained(
                "asafaya/bert-base-arabic"
            )
        elif args["--lang"] == "Spanish":
            self.bert_tokeniser = AutoTokenizer.from_pretrained(
                "dccuchile/bert-base-spanish-wwm-uncased"
            )

        (
            self.inputs,
            self.attention_masks,
            self.lengths,
            self.label_indices,
            self.label_input_ids,
        ) = self.process_data()

    def load_dataset(self):
        """
        :return: dataset after being preprocessed and tokenised
        """
        df = pd.read_csv(self.filename, sep="\t")
        x_train, y_train = df.Tweet.values, df.iloc[:, 2:].values
        # get label names
        label_names = df.columns[2:].tolist()
        return x_train, y_train, label_names

    def process_data(self):
        desc = "PreProcessing dataset {}...".format("")
        preprocessor = twitter_preprocessor()

        if self.args["--lang"] == "English":
            # flat self.label_names
            segment_a = " ".join(self.label_names) + "?"
            print(segment_a)

        inputs, attention_masks, lengths, label_indices, label_input_ids = (
            [],
            [],
            [],
            [],
            [],
        )
        for data_idx, data_item in enumerate(tqdm(self.data, desc=desc)):
            data_item = " ".join(preprocessor(data_item))
            data_item = self.bert_tokeniser.encode_plus(
                segment_a,
                data_item,
                add_special_tokens=True,
                max_length=self.max_length,
                pad_to_max_length=True,
                truncation=True,
            )
            input_id = data_item["input_ids"]
            attention_mask = data_item["attention_mask"]
            input_length = len([i for i in data_item["attention_mask"] if i == 1])
            inputs.append(input_id)
            lengths.append(input_length)
            attention_masks.append(attention_mask)

            # label indices
            label_idxs = [
                self.bert_tokeniser.convert_ids_to_tokens(input_id).index(
                    self.label_names[idx]
                )
                for idx, _ in enumerate(self.label_names)
            ]
            label_indices.append(label_idxs)

            # get label id
            label_input_id = self.bert_tokeniser.encode_plus(
                segment_a,
                add_special_tokens=False,
                max_length=self.max_length,
                pad_to_max_length=True,
                truncation=True,
            )["input_ids"]
            if len(self.all_label_input_ids) == 0:
                self.all_label_input_ids = label_input_id
            # extract label input ids from self.labels[data_idx] if it is not zero
            label_input_id = [
                (
                    label_input_id[label_idxs[idx]]
                    if self.labels[data_idx][idx] == 1
                    else 0
                )
                for idx, _ in enumerate(self.label_names)
            ]
            # remove 0s in label_input_ids
            label_input_id = list(filter(lambda a: a != 0, label_input_ids))
            label_input_ids.append(label_input_id)

        inputs = torch.tensor(inputs, dtype=torch.long)
        data_length = torch.tensor(lengths, dtype=torch.long)
        label_indices = torch.tensor(label_indices, dtype=torch.long)
        attention_masks = torch.tensor(attention_masks, dtype=torch.long)
        label_input_ids = torch.tensor(label_input_ids, dtype=torch.long)
        return inputs, attention_masks, data_length, label_indices, label_input_ids

    def __getitem__(self, index):
        inputs = self.inputs[index]
        labels = self.labels[index]
        label_idxs = self.label_indices[index]
        length = self.lengths[index]
        label_input_ids = self.label_input_ids[index]
        attention_masks = self.attention_masks[index]
        all_label_input_ids = self.all_label_input_ids
        return (
            inputs,
            attention_masks,
            labels,
            length,
            label_idxs,
            label_input_ids,
            all_label_input_ids,
        )

    def __len__(self):
        return len(self.inputs)
