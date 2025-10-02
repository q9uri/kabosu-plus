from kabosu_plus.sbv2.nlp import language_selector
from kabosu_plus.sbv2.constants import Languages
from kabosu_plus import normalize_text

def g2p(text: str,
        raise_yomi_error: bool = False,
        language_list:list[Languages] = [Languages.JP],
        keihan: bool = False,
        babytalk: bool = False, 
        dakuten: bool = False,
        use_jp_extra: bool = False,
        ) -> tuple[Languages, str, list[str], list[int], list[int], list[str] | None, list[str] | None, list[str] | None]:

    if len(language_list) == 1:
        language = language_list[0] 
        if language == Languages.MULTI:
            language = language_selector(text, [Languages.EN, Languages.JP, Languages.ZH, Languages.KO])

    else:
        language = language_selector(text, language_list)
        
   
    if language == Languages.JP:

        from kabosu_plus.sbv2.nlp.japanese import g2p as g2p_ja
        norm_text = normalize_text(text)
        norm_text, phones, tones, word2ph, sep_text, sep_kata ,sep_kata_with_joshi = g2p_ja.g2p(norm_text=norm_text,
                                                                                                use_jp_extra=use_jp_extra,
                                                                                                raise_yomi_error=raise_yomi_error,
                                                                                                keihan=keihan,
                                                                                                babytalk=babytalk,
                                                                                                dakuten=dakuten,                                                                                                
                                                                                                )
        return Languages.JP, norm_text, phones, tones, word2ph, sep_text, sep_kata ,sep_kata_with_joshi

    elif language == Languages.EN:
        from kabosu_plus.sbv2.nlp.english import g2p as g2p_en 

        norm_text, phones, tones, word2ph = g2p_en.g2p(text=text)
        sep_text = None
        sep_kata = None
        sep_kata_with_joshi = None
    
    elif language == Languages.ZH:
        from kabosu_plus.sbv2.nlp.chinese import g2p as g2p_zh

        norm_text, phones, tones, word2ph = g2p_zh.g2p(text=text)
        sep_text = None
        sep_kata = None
        sep_kata_with_joshi = None

    elif language == Languages.KO:
        from kabosu_plus.sbv2.nlp.korean import g2p as g2p_ko
        norm_text, phones, tones, word2ph = g2p_ko.g2p(text=text, raise_yomi_error=raise_yomi_error)
        sep_text = None
        sep_kata = None
        sep_kata_with_joshi = None

    return language, norm_text, phones, tones, word2ph, sep_text, sep_kata ,sep_kata_with_joshi