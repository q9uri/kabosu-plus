from kabosu_plus.sbv2.constants import Languages

from typing import Literal
import re

# ↓ ひらがな、カタカナ、漢字
JAPANESE_PATTERN = re.compile(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF\u3005]+")
CHINESE_PATTERN = re.compile(r"[\u4e00-\u9fa5]+")

HANGUL_SYLLABLES = r"\uac00-\ud7a3"
HANGUL_JAMO = r"\u1100-\u11ff"
HANGUL_COMPTIBILITY_JAMO = r"\u3131-\u318e"
KOREAN_CHAR = HANGUL_SYLLABLES + HANGUL_JAMO + HANGUL_COMPTIBILITY_JAMO

KOREAN_PATTERN = re.compile(rf"[{KOREAN_CHAR}]+")

def language_selector(
        text:str,
        supported_languge:Literal[Languages] = Languages.JP,
        ) -> Languages:
    
    #日文漢字＋中文漢字
    if CHINESE_PATTERN.match(text):
        if "ZH" in supported_languge:
            lang = Languages.ZH
        else:
            lang = Languages.JP

    #日本語かなカナ漢字
    elif JAPANESE_PATTERN.match(text):
        lang = Languages.JP

    elif KOREAN_PATTERN.match(text):
        if "KO" in supported_languge:
            lang = Languages.KO
        else:
            lang = Languages.JP
    
    else:
        if "EN" in supported_languge:
            lang = Languages.EN
        else:
            lang = Languages.JP

    return lang
    

class YomiError(Exception):
    """
    OpenJTalk で、読みが正しく取得できない箇所があるときに発生する例外。
    基本的に「学習の前処理のテキスト処理時」には発生させ、そうでない場合は、
    raise_yomi_error=False にしておいて、この例外を発生させないようにする。
    """