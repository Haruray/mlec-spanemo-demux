"""
Usage:
    main.py [options]

Options:
    -h --help                         show this screen
    --loss-type=<str>                 Which loss to use cross-ent|corr|joint. [default: cross-entropy]
    --max-length=<int>                text length [default: 128]
    --output-dropout=<float>          prob of dropout applied to the output layer [default: 0.1]
    --seed=<int>                      fixed random seed number [default: 42]
    --train-batch-size=<int>          batch size [default: 32]
    --eval-batch-size=<int>           batch size [default: 32]
    --max-epoch=<int>                 max epoch [default: 20]
    --ffn-lr=<float>                  ffn learning rate [default: 0.001]
    --bert-lr=<float>                 bert learning rate [default: 2e-5]
    --lang=<str>                      language choice [default: English]
    --dev-path=<str>                  file path of the dev set [default: '']
    --train-path=<str>                file path of the train set [default: '']
    --alpha-loss=<float>              weight used to balance the loss [default: 0.2]
"""

from MLEC import (
    Trainer,
    SpanEmo,
    DataClass,
    SpanEmoB2B,
    DemuxLite,
    Demux,
    EmoRec,
    DemuxAdv,
    DemuxJoint,
)
from torch.utils.data import DataLoader
import torch
from docopt import docopt
import datetime
import json
import numpy as np

args = docopt(__doc__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if str(device) == "cuda:0":
    print("Currently using GPU: {}".format(device))
    np.random.seed(int(args["--seed"]))
    torch.cuda.manual_seed_all(int(args["--seed"]))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    print("Currently using CPU")
#####################################################################
# Save hyper-parameter values ---> config.json
# Save model weights ---> filename.pt using current time
#####################################################################
now = datetime.datetime.now()
filename = now.strftime("%Y-%m-%d %H-%M-%S")
fw = open("configs/" + filename + ".json", "a")
model_path = filename + ".pt"
args["--checkpoint-path"] = model_path
json.dump(args, fw, sort_keys=True, indent=2)
#####################################################################
# Define Dataloaders
#####################################################################
args["--lang"] = "Indonesia"
train_dataset = DataClass(args, args["--train-path"])
train_data_loader = DataLoader(
    train_dataset, batch_size=int(args["--train-batch-size"]), shuffle=True
)
print("The number of training batches: ", len(train_data_loader))
# dev_dataset = DataClass(args, args["--dev-path"])
# dev_data_loader = DataLoader(
#     dev_dataset, batch_size=int(args["--eval-batch-size"]), shuffle=False
# )
# print("The number of validation batches: ", len(dev_data_loader))
#############################################################################
# Define Model & Training Pipeline
#############################################################################
label_size = len(train_dataset.label_names)
# model = SpanEmo(
#     output_dropout=float(args["--output-dropout"]),
#     lang=args["--lang"],
#     embedding_vocab_size=len(train_dataset.bert_tokeniser),
# )

# another_model = SpanEmoB2B(
#     output_dropout=float(args["--output-dropout"]),
#     embedding_vocab_size=len(train_dataset.bert_tokeniser),
# )

#############################################################################
# Start Training
#############################################################################
# learn = Trainer(
#     another_model,
#     train_data_loader,
#     dev_data_loader,
#     filename=filename,
#     col_names=train_dataset.label_names,
# )
# learn.fit(num_epochs=int(args["--max-epoch"]), args=args, device=device)

text = ["joy sad angry? I am happy, kind of"] * 2
token = train_dataset.bert_tokeniser(text)
# print tokenized text
demux = Demux(
    output_dropout=float(args["--output-dropout"]),
    lang="Indonesia",
    embedding_vocab_size=len(train_dataset.bert_tokeniser),
    alpha=0,
    beta=0,
    label_size=3,
    device=device,
)
demuxadv = DemuxAdv(
    output_dropout=float(args["--output-dropout"]),
    lang="Indonesia",
    embedding_vocab_size=len(train_dataset.bert_tokeniser),
    alpha=0,
    beta=0,
    label_size=3,
    device=device,
)
demuxjoint = DemuxJoint(
    output_dropout=float(args["--output-dropout"]),
    lang="Indonesia",
    embedding_vocab_size=len(train_dataset.bert_tokeniser),
    alpha=0,
    beta=0,
    label_size=3,
    device=device,
)
spanemo = SpanEmo(
    output_dropout=float(args["--output-dropout"]),
    lang="Indonesia",
    embedding_vocab_size=len(train_dataset.bert_tokeniser),
    alpha=0,
    beta=0,
    device=device,
)
emorec = EmoRec(
    output_dropout=float(args["--output-dropout"]),
    lang="Indonesia",
    embedding_vocab_size=len(train_dataset.bert_tokeniser),
    alpha=0,
    beta=0,
    device=device,
    label_size=3,
)
# output = demux(
#     input_ids=token["input_ids"],
#     input_attention_masks=token["attention_mask"],
#     label_idxs=torch.tensor([1, 2]),
# )
output = demuxjoint(
    input_ids=torch.tensor(token["input_ids"]),
    input_attention_masks=torch.tensor(token["attention_mask"]),
    label_idxs=torch.tensor([[1, 2, 3]]),
)
print(output)
