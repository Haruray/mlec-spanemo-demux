import numpy as np

PLUTCHIK_WHEEL_ANGLES = {
    emo: np.pi / 8 * i
    for i, emo in enumerate(
        [
            "joy",
            "love",
            "trust",
            "submission",
            "fear",
            "awe",
            "surprise",
            "pessimism",
            "sadness",
            "remorse",
            "disgust",
            "contempt",
            "anger",
            "aggression",
            "anticipation",
            "optimism",
        ]
    )
}
