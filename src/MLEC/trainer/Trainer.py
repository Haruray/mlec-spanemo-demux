from fastprogress.fastprogress import format_time, master_bar, progress_bar
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, jaccard_score, hamming_loss
import torch.nn.functional as F
import numpy as np
import torch
import time
from MLEC.trainer.EarlyStopping import EarlyStopping
from MLEC.loss.inter_corr_loss import inter_corr_loss
from MLEC.loss.intra_corr_loss import intra_corr_loss
from MLEC.enums.CorrelationType import CorrelationType
from MLEC.emotion_corr_weightings.Correlations import Correlations
from torch.cuda.amp import GradScaler, autocast
import torch.cuda


class Trainer(object):
    """
    Class to encapsulate training and validation steps for a pipeline. Based off the "Tonks Library"
    :param model: PyTorch model to use with the Learner
    :param train_data_loader: dataloader for all of the training data
    :param val_data_loader: dataloader for all of the validation data
    :param filename: the best model will be saved using this given name (str)
    """

    def __init__(
        self,
        model,
        train_data_loader,
        val_data_loader,
        filename,
        col_names=[],
        corr_type: CorrelationType = CorrelationType.IDENTITY,
    ):
        self.model = model
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.filename = filename
        self.early_stop = EarlyStopping(self.filename, patience=10)
        self.correlations = Correlations(corr_type=corr_type, col_names=col_names)
        self.col_names = col_names

    def fit(self, num_epochs, args, device="cuda:0"):
        """
        Fit the PyTorch model
        :param num_epochs: number of epochs to train (int)
        :param args:
        :param device: str (defaults to 'cuda:0')
        """
        optimizer, scheduler, step_scheduler_on_batch = self.optimizer(args)
        self.model = self.model.to(device)
        scaler = GradScaler()  # Initialize GradScaler
        pbar = master_bar(range(num_epochs))
        headers = [
            "Train_Loss",
            "Val_Loss",
            "Train Inter loss",
            "Train Intra loss",
            "F1-Macro",
            "F1-Micro",
            "JS",
            "Hamming Loss",
            "Time",
        ]
        pbar.write(headers, table=True)
        for epoch in pbar:
            epoch += 1
            start_time = time.time()
            self.model.train()
            overall_training_loss = 0.0
            overall_inter_loss = 0.0
            overall_intra_loss = 0.0
            for step, batch in enumerate(
                progress_bar(self.train_data_loader, parent=pbar)
            ):
                optimizer.zero_grad()
                with autocast():  # Enable autocast
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

                    num_rows, _, logits, targets, last_hidden_state = self.model(
                        input_ids=inputs,
                        input_attention_masks=attention_masks,
                        targets=targets,
                        target_input_ids=label_input_ids,
                        target_attention_masks=label_attention_masks,
                        lengths=lengths,
                        label_idxs=label_idxs,
                        all_label_input_ids=all_label_input_ids,
                    )
                    inter_corr_loss_total = intra_corr_loss(
                        logits, targets, self.correlations
                    )
                    intra_corr_loss_total = inter_corr_loss(
                        logits, targets, self.correlations
                    )
                    bce_loss = F.binary_cross_entropy_with_logits(logits, targets).to(
                        device
                    )
                    targets = targets.cpu().numpy()
                    total_loss = (
                        bce_loss * (1 - (self.model.alpha + self.model.beta))
                        + (inter_corr_loss_total * self.model.alpha)
                        + (intra_corr_loss_total * self.model.beta)
                    )
                overall_training_loss += total_loss.item() * num_rows
                overall_inter_loss += inter_corr_loss_total.item() * num_rows
                overall_intra_loss += intra_corr_loss_total.item() * num_rows
                scaler.scale(total_loss).backward()  # Scale the loss value
                # total_loss.backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                # optimizer.step()
                scaler.step(
                    optimizer
                )  # Unscales the gradients and calls optimizer.step()
                scaler.update()  # Updates the scale for next iteration
                if step_scheduler_on_batch:
                    scheduler.step()

                # Free unused GPU memory
                torch.cuda.empty_cache()

            if not step_scheduler_on_batch:
                scheduler.step()

            overall_training_loss = overall_training_loss / len(
                self.train_data_loader.dataset
            )
            overall_training_inter_loss = overall_inter_loss / len(
                self.train_data_loader.dataset
            )
            overall_training_intra_loss = overall_intra_loss / len(
                self.train_data_loader.dataset
            )

            overall_val_loss, pred_dict = self.predict(device, pbar)
            y_true, y_pred = pred_dict["y_true"], pred_dict["y_pred"]

            str_stats = []
            stats = [
                overall_training_loss,
                overall_val_loss,
                overall_training_inter_loss,
                overall_training_intra_loss,
                f1_score(y_true, y_pred, average="macro", zero_division=1),
                f1_score(y_true, y_pred, average="micro", zero_division=1),
                jaccard_score(y_true, y_pred, average="samples", zero_division=1),
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
                {"params": self.model.encoder_parameters},
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
        scaler = GradScaler()  # Initialize GradScaler
        print("len col names: ", len(self.col_names))
        preds_dict = {
            "y_true": np.zeros([current_size, len(self.col_names)]),
            "y_pred": np.zeros([current_size, len(self.col_names)]),
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
                with autocast():  # Enable autocast
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

                    num_rows, y_pred, logits, targets, last_hidden_state = self.model(
                        input_ids=inputs,
                        input_attention_masks=attention_masks,
                        targets=targets,
                        target_input_ids=inputs,
                        target_attention_masks=attention_masks,
                        lengths=lengths,
                        label_idxs=label_idxs,
                        all_label_input_ids=all_label_input_ids,
                    )
                    inter_corr_loss_total = intra_corr_loss(
                        logits, targets, self.correlations
                    )
                    intra_corr_loss_total = inter_corr_loss(
                        logits, targets, self.correlations
                    )
                    bce_loss = F.binary_cross_entropy_with_logits(logits, targets).to(
                        device
                    )
                    targets = targets.cpu().numpy()
                    total_loss = (
                        bce_loss * (1 - (self.model.alpha + self.model.beta))
                        + (inter_corr_loss_total * self.model.alpha)
                        + (intra_corr_loss_total * self.model.beta)
                    )
                overall_val_loss += total_loss.item() * num_rows

                current_index = index_dict
                preds_dict["y_true"][
                    current_index : current_index + num_rows, :
                ] = targets
                preds_dict["y_pred"][
                    current_index : current_index + num_rows, :
                ] = y_pred
                index_dict += num_rows
                # Free unused GPU memory
                torch.cuda.empty_cache()

        overall_val_loss = overall_val_loss / len(self.val_data_loader.dataset)
        return overall_val_loss, preds_dict
