
from .types import NjdObject
from .sbv2.nlp.japanese.normalizer import normalize_text as normalize_text_plus
from typing import Union
from pathlib import Path
from kabosu_core import pyopenjtalk
from kabosu_core.pyopenjtalk.normalizer import normalize_text as normalize_text_core
import jpreprocess

def load_marine_model(model_dir: Union[str, None] = None, dict_dir: Union[str, None] = None):
    pyopenjtalk.load_marine_model(model_dir=model_dir, dict_dir=dict_dir)

def update_global_jtalk_with_user_dict(
        user_dictionary: str | Path | None = None
        ) -> None:
    """Update global openjtalk instance with the user dictionary

    Note that this will change the global state of the openjtalk module.

    """
    pyopenjtalk.update_global_jtalk_with_user_dict(user_dictionary=user_dictionary)

def extract_fullcontext(
        text: str,
        hankaku: bool = True,
        itaiji: bool = True,
        kanalizer: bool = False,
        yomikata: bool = True,
        sbv2: bool = True,
        use_vanilla: bool = False,
        use_ko2ja: bool = True,
        run_marine: bool = False,
        keihan: bool = False,
        babytalk: bool = False,
        dakuten: bool = False,
        jpreprocess: Union[jpreprocess.JPreprocess, None] = None
    ) -> list[str]:
    
    """
    ### input 
    text (str): input text  
    ## output
    => list[str] : fullcontext label
    """
    text = normalize_text(
        text=text,
        hankaku=hankaku,
        itaiji=itaiji,
        kanalizer=kanalizer,
        yomikata=yomikata,
        sbv2=sbv2
    )

    return pyopenjtalk.extract_fullcontext(
        text=text,
        hankaku=hankaku,
        itaiji=itaiji,
        yomikata=yomikata,
        kanalizer=kanalizer,
        use_vanilla=use_vanilla,
        use_ko2ja=use_ko2ja,
        run_marine=run_marine,
        keihan=keihan,
        babytalk=babytalk,
        dakuten=dakuten,
        jpreprocess=jpreprocess
        )


def g2p(
        text: str,
        hankaku: bool = True,
        itaiji: bool = True,
        kanalizer: bool = False,
        yomikata: bool = True,
        sbv2: bool = True,
        use_vanilla: bool = False,
        use_ko2ja: bool = True,
        run_marine: bool = False,
        keihan: bool = False,
        babytalk: bool = False,
        dakuten: bool = False,
        kana: bool = False,
        join: bool = True,
        jpreprocess: Union[jpreprocess.JPreprocess, None] = None
    ):
    text = normalize_text(
        text=text,
        hankaku=hankaku,
        itaiji=itaiji,
        kanalizer=kanalizer,
        yomikata=yomikata,
        sbv2=sbv2
    )
        
    return pyopenjtalk.g2p(
        text=text,
        hankaku=False,
        itaiji=False,
        kanalizer=False,
        yomikata=False,
        use_vanilla=use_vanilla,
        use_ko2ja=use_ko2ja,
        run_marine=run_marine,
        keihan=keihan,
        babytalk=babytalk,
        dakuten=dakuten,
        kana=kana,
        join=join,
        jpreprocess=jpreprocess
    )

def run_frontend(
        text: str,
        hankaku: bool = True,
        itaiji: bool = True,
        kanalizer: bool = False,
        yomikata: bool = True,
        sbv2: bool = True,
        use_vanilla: bool = False,
        use_ko2ja: bool = True,
        run_marine: bool = False,
        keihan: bool = False,
        babytalk: bool = False,
        dakuten: bool = False,
        jpreprocess: Union[jpreprocess.JPreprocess, None] = None
    ) -> list[NjdObject]:
    """
    ### input 
    text (str): input text  
    hankaku (bool) : True:
    itaiji: (bool) : True:
    kanalizer: (bool) : True:
    yomikata: (bool) : True:
    kanji_yomi: (bool) : True:
    process_odori (bool) : True:
    ## output
    => list[NjdObject] : njd_features
    """
    text = normalize_text(
        text=text,
        hankaku=hankaku,
        itaiji=itaiji,
        kanalizer=kanalizer,
        yomikata=yomikata,
        sbv2=sbv2
    )

    return pyopenjtalk.run_frontend(
        text=text,
        hankaku=False,
        itaiji=False,
        kanalizer=False,
        yomikata=False,
        keihan=keihan,
        babytalk=babytalk,
        dakuten=dakuten,
        use_vanilla=use_vanilla,
        use_ko2ja=use_ko2ja,
        run_marine=run_marine,
        jpreprocess=jpreprocess
    )

def make_label(
        njd_features: list[NjdObject], 
        jpreprocess: Union[jpreprocess.JPreprocess, None] = None
        ) -> list[str]:
    
    return pyopenjtalk.make_label(
        njd_features=njd_features, 
        jpreprocess=jpreprocess
        )


def reader_furigana(text:str):
    # this liblary use fork version yomikata
    # https://github.com/q9uri/yomikata

    return pyopenjtalk.reader_furigana(text=text)

def dictreader_furigana(text:str):
    return pyopenjtalk.dictreader_furigana(text=text)


def kanalizer_convert(text: str):
    return pyopenjtalk.kanalizer_convert(text=text)

def normalize_text(
        text: str,
        hankaku: bool = True,
        itaiji: bool = True,
        kanalizer: bool = True,
        yomikata: bool = True,
        sbv2:bool = True
    ) -> str:
    """
    ### input 
    text (str): input text  
    hankaku (bool): convert hankaku to zenkaku  
    itaiji (bool): convert itaiji to joyo-kanji
    ## output
    str : normalized text
    """
    if sbv2:
        text = normalize_text_plus(text)

    return  normalize_text_core(
        text=text,
        hankaku=hankaku,
        itaiji=itaiji,
        kanalizer=kanalizer,
        yomikata=yomikata
    ) 


