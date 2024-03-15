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
        },
        all_caps_tag="wrap",
        fix_text=False,
        segmenter="corpus_id_web_wiki",
        corrector="corpus_id_web_wiki",
        unpack_hashtags=True,
        unpack_contractions=True,
        spell_correct_elong=False,
        demojize=True,
        demojize_lang="id",
        tokenizer=SocialTokenizer(lowercase=True).tokenize,
    ).pre_process_doc
    return preprocessor
