from MLEC import Correlations, CorrelationType
import numpy as np

corr = Correlations(
    col_names=[
        "anger",
        "anticipation",
        "disgust",
        "fear",
        "joy",
        "sadness",
        "surprise",
        "trust",
    ],
    corr_type=CorrelationType.PLUTCHIK,
)

col_names = np.array(
    [
        "anger",
        "anticipation",
        "disgust",
        "fear",
        "joy",
        "sadness",
        "surprise",
        "trust",
    ]
)

idx_1 = [1]
idx_2 = [2, 3]

print(
    corr.get(
        index=(
            (list(col_names[idx_1])),
            (list(col_names[idx_2])),
        ),
        decreasing=True,
    )
)
