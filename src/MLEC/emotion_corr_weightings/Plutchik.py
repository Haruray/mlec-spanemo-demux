import numpy as np

PLUTCHIK_WHEEL_ANGLES = {
    emo: np.pi / 4 * i
    for i, emo in enumerate(
        [
            "joy",
            "trust",
            "fear",
            "surprise",
            "sadness",
            "disgust",
            "anger",
            "anticipation",
        ]
    )
}
