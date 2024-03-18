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
            self.bert_tokeniser.add_tokens(self.label_names)
        elif args["--lang"] == "Indonesia":
            self.bert_tokeniser = AutoTokenizer.from_pretrained(
                "indolem/indobert-base-uncased"
            )
            self.bert_tokeniser.add_tokens(self.label_names)

        (
            self.inputs,
            self.attention_masks,
            self.lengths,
            self.label_indices,
            self.label_input_ids,
            self.label_attention_masks,
        ) = self.process_data()

    def load_dataset(self):
        """
        :return: dataset after being preprocessed and tokenised
        """
        # if the file is a text file, then use sep="\t"
        if self.filename.endswith(".txt"):
            df = pd.read_csv(self.filename, sep="\t")
        else:
            # if the file is a csv file, then use sep=","
            df = pd.read_csv(self.filename, sep=",")
        x_train, y_train = df.Tweet.values, df.iloc[:, 2:].values
        # get label names
        label_names = df.columns[2:].tolist()
        return x_train, y_train, label_names

    def process_data(self):
        desc = "PreProcessing dataset {}...".format("")
        preprocessor = twitter_preprocessor(lang=self.args["--lang"])

        if self.args["--lang"] == "English" or self.args["--lang"] == "Indonesia":
            # flat self.label_names
            segment_a = " ".join(self.label_names) + "?"

        (
            inputs,
            attention_masks,
            lengths,
            label_indices,
            label_input_ids,
            label_attention_masks,
        ) = ([], [], [], [], [], [])

        self.all_label_input_ids = [
            self.bert_tokeniser.encode(label_name, add_special_tokens=False)
            for label_name in self.label_names
        ]
        self.all_label_input_ids = torch.tensor(
            self.all_label_input_ids, dtype=torch.long
        )

        for data_idx, data_item in enumerate(tqdm(self.data, desc=desc)):
            data_item = " ".join(preprocessor(data_item))
            data_item = self.bert_tokeniser.encode_plus(
                segment_a,
                data_item,
                add_special_tokens=True,
                max_length=self.max_length,
                padding="max_length",
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

            # get target label names
            current_target_label = self.labels[data_idx]
            current_target_label_names = [
                self.label_names[idx]
                for idx, val in enumerate(current_target_label)
                if val == 1
            ]
            # print(current_target_label_names)
            # input_ids and attention_masks for the target labels
            label_input_id = self.bert_tokeniser.encode_plus(
                " ".join(current_target_label_names),
                add_special_tokens=True,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
            )
            label_input_ids.append(label_input_id["input_ids"])
            label_attention_masks.append(label_input_id["attention_mask"])
        inputs = torch.tensor(inputs, dtype=torch.long)
        data_length = torch.tensor(lengths, dtype=torch.long)
        label_indices = torch.tensor(label_indices, dtype=torch.long)
        attention_masks = torch.tensor(attention_masks, dtype=torch.long)
        label_input_ids = torch.tensor(label_input_ids, dtype=torch.long)
        label_attention_masks = torch.tensor(label_attention_masks, dtype=torch.long)
        return (
            inputs,
            attention_masks,
            data_length,
            label_indices,
            label_input_ids,
            label_attention_masks,
        )

    def __getitem__(self, index):
        inputs = self.inputs[index]
        labels = self.labels[index]
        label_idxs = self.label_indices[index]
        length = self.lengths[index]
        label_input_ids = self.label_input_ids[index]
        attention_masks = self.attention_masks[index]
        label_attention_masks = self.label_attention_masks[index]
        all_label_input_ids = self.all_label_input_ids
        return (
            inputs,
            attention_masks,
            labels,
            length,
            label_idxs,
            label_input_ids,
            label_attention_masks,
            all_label_input_ids,
        )

    def __len__(self):
        return len(self.inputs)
