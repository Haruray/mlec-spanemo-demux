from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.classes.preprocessor import TextPreProcessor


def twitter_preprocessor(lang="English", demojize=True):
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
        segmenter="corpus_id_web_wiki" if lang == "Indonesia" else "twitter_2018",
        corrector="corpus_id_web_wiki" if lang == "Indonesia" else "twitter_2018",
        unpack_hashtags=True,
        unpack_contractions=True,
        spell_correct_elong=False,
        demojize=demojize,
        demojize_lang="id" if lang == "Indonesia" else "en",
        tokenizer=SocialTokenizer(lowercase=True).tokenize,
    ).pre_process_doc
    return preprocessor
