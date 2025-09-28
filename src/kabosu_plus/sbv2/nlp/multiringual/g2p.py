from kabosu_plus.sbv2.nlp import languge_selector
from kabosu_plus.sbv2.constants import Languages


def g2p(text: str,
        keihan: bool = False,
        babytalk: bool = False, 
        dakuten: bool = False,
        ) -> tuple[Languages, list[str], list[int], list[int], list[str] | None, list[str] | None, list[str] | None]:

    language = languge_selector(text, supported_languge=[Languages.EN, Languages.JP, Languages.ZH])
    
    if language == Languages.JP:
        from kabosu_plus.sbv2.nlp.japanese import g2p as g2p_ja
        phones, tones, word2ph, sep_text, sep_kata ,sep_kata_with_joshi = g2p_ja.g2p(norm_text=text, keihan=keihan, babytalk=babytalk, dakuten=dakuten)
        return Languages.JP, phones, tones, word2ph, sep_text, sep_kata ,sep_kata_with_joshi

    elif language == Languages.EN:
        from kabosu_plus.sbv2.nlp.english import normalizer  
        from kabosu_plus.sbv2.nlp.english import g2p as g2p_en 
        norm_text = normalizer.normalize_text(text)
        phones, tones, word2ph = g2p_en.g2p(text=norm_text)
        return Languages.EN, phones, tones, word2ph, None, None, None
    
    elif language == Languages.ZH:
        from kabosu_plus.sbv2.nlp.chinese import g2p as g2p_zh
        phones, tones, word2ph = g2p_zh.g2p(text=text)

        return Languages.ZH, phones, tones, word2ph, None, None, None