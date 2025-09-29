from kabosu_plus.sbv2.nlp import languge_selector
from kabosu_plus.sbv2.constants import Languages

def g2p(text: str,
        keihan: bool = False,
        babytalk: bool = False, 
        dakuten: bool = False,
        language_list:list[Languages] = [Languages.JP],
        ) -> tuple[Languages, list[str], list[int], list[int], list[str] | None, list[str] | None, list[str] | None]:

    if len(language_list) == 1:
        language = language_list[0] 
        if language == Languages.MULTI:
            language = languge_selector(text, [Languages.EN, Languages.JP, Languages.ZH])

    else:
        language = languge_selector(text, language_list)
        
   
    if language == Languages.JP:

        from kabosu_plus.sbv2.nlp.japanese import g2p as g2p_ja
        phones, tones, word2ph, sep_text, sep_kata ,sep_kata_with_joshi = g2p_ja.g2p(norm_text=text, keihan=keihan, babytalk=babytalk, dakuten=dakuten)
        return Languages.JP, phones, tones, word2ph, sep_text, sep_kata ,sep_kata_with_joshi

    elif language == Languages.EN:
        from kabosu_plus.sbv2.nlp.english import g2p as g2p_en 

        phones, tones, word2ph = g2p_en.g2p(text=text)
        return Languages.EN, phones, tones, word2ph, None, None, None
    
    elif language == Languages.ZH:
        from kabosu_plus.sbv2.nlp.chinese import g2p as g2p_zh

        phones, tones, word2ph = g2p_zh.g2p(text=text)

        return Languages.ZH, norm_text, phones, tones, word2ph, None, None, None