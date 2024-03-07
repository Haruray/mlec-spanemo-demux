from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.classes.preprocessor import TextPreProcessor


def twitter_preprocessor():
    preprocessor = TextPreProcessor(
        normalize=["url", "email", "phone", "user"],
        annotate={
            "hashtag",
            "elongated",
            "allcaps",
            "repeated",
            "emphasis",
            "censored",
        },
        all_caps_tag="wrap",
        fix_text=False,
        segmenter="twitter_2018",
        corrector="twitter_2018",
        unpack_hashtags=True,
        unpack_contractions=True,
        spell_correct_elong=False,
        tokenizer=SocialTokenizer(lowercase=True).tokenize,
    ).pre_process_doc
    return preprocessor
