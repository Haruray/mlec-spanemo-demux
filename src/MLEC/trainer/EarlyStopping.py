from fastprogress.fastprogress import format_time, master_bar, progress_bar
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, jaccard_score
import torch.nn.functional as F
import numpy as np
import torch
import time


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.
    Taken from https://github.com/Bjarten/early-stopping-pytorch"""

    def __init__(
        self,
        filename,
        patience=7,
        verbose=True,
        delta=0,
        criteria="val_loss",
        bigger_better=False,
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.cur_date = filename
        self.criteria = criteria
        self.bigger_better = bigger_better

    def __call__(self, criteria_score, model):
        if self.best_score is None:
            self.best_score = criteria_score
            self.save_checkpoint(model)
        elif criteria_score - self.best_score <= -self.delta and self.bigger_better:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        elif self.best_score - criteria_score <= -self.delta and not self.bigger_better:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            prev_best = self.best_score
            is_delta_tolerated = (
                criteria_score < self.best_score
                if self.bigger_better
                else criteria_score > self.best_score
            )
            self.best_score = self.best_score if is_delta_tolerated else criteria_score
            self.save_checkpoint(
                model,
                delta_tolerated=is_delta_tolerated,
                prev_best=prev_best,
            )
            self.counter = 0

    def save_checkpoint(self, model, delta_tolerated=False, prev_best=np.inf):
        """Saves model when validation loss decrease."""
        status = "increased" if self.bigger_better else "decreased"
        if self.verbose:
            if delta_tolerated:
                print("Delta tolerated. Saving model ...")
            else:
                print(
                    f"{self.criteria} {status} ({prev_best:.6f} --> {self.best_score:.6f}).  Saving model ..."
                )
        torch.save(model.state_dict(), "../models/" + self.cur_date + "_checkpoint.pt")
