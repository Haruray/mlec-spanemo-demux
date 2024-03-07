from fastprogress.fastprogress import format_time, master_bar, progress_bar
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, jaccard_score, hamming_loss
import torch.nn.functional as F
import numpy as np
import torch
import time
from MLEC.trainer.EarlyStopping import EarlyStopping


class Trainer(object):
    """
    Class to encapsulate training and validation steps for a pipeline. Based off the "Tonks Library"
    :param model: PyTorch model to use with the Learner
    :param train_data_loader: dataloader for all of the training data
    :param val_data_loader: dataloader for all of the validation data
    :param filename: the best model will be saved using this given name (str)
    """

    def __init__(self, model, train_data_loader, val_data_loader, filename):
        self.model = model
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.filename = filename
        self.early_stop = EarlyStopping(self.filename, patience=10)

    def fit(self, num_epochs, args, device="cuda:0"):
        """
        Fit the PyTorch model
        :param num_epochs: number of epochs to train (int)
        :param args:
        :param device: str (defaults to 'cuda:0')
        """
        optimizer, scheduler, step_scheduler_on_batch = self.optimizer(args)
        self.model = self.model.to(device)
        pbar = master_bar(range(num_epochs))
        headers = ["Train_Loss", "Val_Loss", "F1-Macro", "F1-Micro", "JS", "Time"]
        pbar.write(headers, table=True)
        for epoch in pbar:
            epoch += 1
            start_time = time.time()
            self.model.train()
            overall_training_loss = 0.0
            for step, batch in enumerate(
                progress_bar(self.train_data_loader, parent=pbar)
            ):
                loss, num_rows, _, _ = self.model(batch, device)
                overall_training_loss += loss.item() * num_rows

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                if step_scheduler_on_batch:
                    scheduler.step()
                optimizer.zero_grad()

            if not step_scheduler_on_batch:
                scheduler.step()

            overall_training_loss = overall_training_loss / len(
                self.train_data_loader.dataset
            )
            overall_val_loss, pred_dict = self.predict(device, pbar)
            y_true, y_pred = pred_dict["y_true"], pred_dict["y_pred"]

            str_stats = []
            stats = [
                overall_training_loss,
                overall_val_loss,
                f1_score(y_true, y_pred, average="macro"),
                f1_score(y_true, y_pred, average="micro"),
                jaccard_score(y_true, y_pred, average="samples"),
                hamming_loss(y_true, y_pred),
            ]

            for stat in stats:
                str_stats.append(
                    "NA"
                    if stat is None
                    else str(stat) if isinstance(stat, int) else f"{stat:.4f}"
                )
            str_stats.append(format_time(time.time() - start_time))
            print("epoch#: ", epoch)
            pbar.write(str_stats, table=True)
            self.early_stop(overall_val_loss, self.model)
            if self.early_stop.early_stop:
                print("Early stopping")
                break

    def optimizer(self, args):
        """

        :param args: object
        """
        optimizer = AdamW(
            [
                {"params": self.model.bert.parameters()},
                {"params": self.model.ffn.parameters(), "lr": float(args["--ffn-lr"])},
            ],
            lr=float(args["--bert-lr"]),
            correct_bias=True,
        )
        num_train_steps = (
            int(len(self.train_data_loader.dataset)) / int(args["--train-batch-size"])
        ) * int(args["--max-epoch"])
        num_warmup_steps = int(num_train_steps * 0.1)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps,
        )
        step_scheduler_on_batch = True
        return optimizer, scheduler, step_scheduler_on_batch

    def predict(self, device="cuda:0", pbar=None):
        """
        Evaluate the model on a validation set
        :param device: str (defaults to 'cuda:0')
        :param pbar: fast_progress progress bar (defaults to None)
        :returns: overall_val_loss (float), accuracies (dict{'acc': value}, preds (dict)
        """
        current_size = len(self.val_data_loader.dataset)
        preds_dict = {
            "y_true": np.zeros([current_size, 11]),
            "y_pred": np.zeros([current_size, 11]),
        }
        overall_val_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            index_dict = 0
            for step, batch in enumerate(
                progress_bar(
                    self.val_data_loader, parent=pbar, leave=(pbar is not None)
                )
            ):
                loss, num_rows, y_pred, targets = self.model(batch, device)
                overall_val_loss += loss.item() * num_rows

                current_index = index_dict
                preds_dict["y_true"][
                    current_index : current_index + num_rows, :
                ] = targets
                preds_dict["y_pred"][
                    current_index : current_index + num_rows, :
                ] = y_pred
                index_dict += num_rows

        overall_val_loss = overall_val_loss / len(self.val_data_loader.dataset)
        return overall_val_loss, preds_dict
