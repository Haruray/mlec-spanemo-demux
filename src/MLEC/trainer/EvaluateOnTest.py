from fastprogress.fastprogress import format_time, master_bar, progress_bar
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, jaccard_score
import torch.nn.functional as F
import numpy as np
import torch
import time


class EvaluateOnTest(object):
    """
    Class to encapsulate evaluation on the test set. Based off the "Tonks Library"
    :param model: PyTorch model to use with the Learner
    :param test_data_loader: dataloader for all of the validation data
    :param model_path: path of the trained model
    """

    def __init__(self, model, test_data_loader, model_path):
        self.model = model
        self.test_data_loader = test_data_loader
        self.model_path = model_path

    def predict(self, device="cuda:0", pbar=None):
        """
        Evaluate the model on a validation set
        :param device: str (defaults to 'cuda:0')
        :param pbar: fast_progress progress bar (defaults to None)
        :returns: None
        """
        self.model.to(device).load_state_dict(torch.load(self.model_path))
        self.model.eval()
        current_size = len(self.test_data_loader.dataset)
        preds_dict = {
            "y_true": np.zeros([current_size, 11]),
            "y_pred": np.zeros([current_size, 11]),
        }
        start_time = time.time()
        with torch.no_grad():
            index_dict = 0
            for step, batch in enumerate(
                progress_bar(
                    self.test_data_loader, parent=pbar, leave=(pbar is not None)
                )
            ):
                _, num_rows, y_pred, targets = self.model(batch, device)
                current_index = index_dict
                preds_dict["y_true"][
                    current_index : current_index + num_rows, :
                ] = targets
                preds_dict["y_pred"][
                    current_index : current_index + num_rows, :
                ] = y_pred
                index_dict += num_rows

        y_true, y_pred = preds_dict["y_true"], preds_dict["y_pred"]
        str_stats = []
        stats = [
            f1_score(y_true, y_pred, average="macro"),
            f1_score(y_true, y_pred, average="micro"),
            jaccard_score(y_true, y_pred, average="samples"),
        ]

        for stat in stats:
            str_stats.append(
                "NA"
                if stat is None
                else str(stat) if isinstance(stat, int) else f"{stat:.4f}"
            )
        str_stats.append(format_time(time.time() - start_time))
        headers = ["F1-Macro", "F1-Micro", "JS", "Time"]
        print(" ".join("{}: {}".format(*k) for k in zip(headers, str_stats)))
