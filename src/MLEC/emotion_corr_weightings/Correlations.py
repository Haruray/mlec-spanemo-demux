from typing import Optional, Any, Dict, Iterable, Union, List, Tuple

import torch
import numpy as np
from MLEC.emotion_corr_weightings.Plutchik import PLUTCHIK_WHEEL_ANGLES
from MLEC.enums.CorrelationType import CorrelationType


class Correlations:
    """Wrapper around correlation matrix.

    Attributes:
        col_names: names of variables.
        func: function of correlation to return.
        corrs: correlation tensor.
        active: whether this class is active.
    """

    def __init__(
        self,
        corr_type: CorrelationType = CorrelationType.IDENTITY,
        col_names: List[str] = [],
        normalize: bool = True,
        active: bool = True,
    ):
        """Init.

        Args:
            batched_data: input matrix, each variable is a column.
            col_names: names of columns, aka variables.
            func: function of correlation to use,
                default is decreasing identity.
            normalize: whether to project correlations to [0, 1].
            active: whether this module is active.
        """

        self.active = active
        self.corr_type = corr_type

        if active:

            assert col_names is not None

            self.col_names = col_names

            # Set correlation matrix
            self.corrs = torch.tensor(
                [
                    [
                        (
                            np.cos(
                                PLUTCHIK_WHEEL_ANGLES[col_i]
                                - PLUTCHIK_WHEEL_ANGLES[col_j]
                            )
                            if self.corr_type == CorrelationType.PLUTCHIK
                            else 1
                        )
                        for col_j in self.col_names
                    ]
                    for col_i in self.col_names
                ]
            )

        if normalize:
            self.corrs = self.corrs / 2 + 0.5

    def _handle_index(
        self, idx: Union[int, List[int], str, List[str]]
    ) -> Optional[List[int]]:
        """Appropriately transforms each dimensions index/ices
        to a list of ints.

        Args:
            idx: index/indices of one dimension.

        Returns:
            List of integers indices.
        """

        def rec_list_elem(l):
            if not l:
                return
            if isinstance(l, list):
                return rec_list_elem(l[0])
            return l

        if hasattr(idx, "__len__") and not idx:
            return

        if isinstance(idx, (str, int)):
            idx = [idx]

        if isinstance(rec_list_elem(idx), str):
            assert (
                self.col_names is not None
            ), "str indexing only if column names provided"
            idx = [self.col_names.index(i) for i in idx]

        return idx

    def get(
        self,
        index: Tuple[Union[int, List[int], str, List[str]]],
        decreasing: bool = False,
    ) -> Optional[float]:
        """Returns correlation between variables. If multiple indices are
        provided for both dims, then all pairs are returned (inner loop is
        first dim). If not `active`, returns nothing."""

        if not self.active:
            return
        _is, js = index
        _is, js = self._handle_index(_is), self._handle_index(js)

        if _is is None or js is None:
            return
        # if decreasing, then return 1 - correlation
        if decreasing:
            return torch.tensor(
                [[1 - self.corrs[i, j] for i in _is] for j in js]
            ).squeeze()
        else:
            return torch.tensor([[self.corrs[i, j] for i in _is] for j in js]).squeeze()
