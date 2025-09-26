from kabosu_plus.sbv2.constants import Languages

from typing import Literal

# ↓ ひらがな、カタカナ、漢字
JAPANESE_PATTERN = re.compile(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF\u3005]+")
CHINESE_PATTERN = re.compile(r"[\u4e00-\u9fa5]+")

def languge_selector(
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

    else:
        if "EN" in supported_languge:
            lang = Languages.EN
        else:
            lang = Languages.JP

    return lang
    