
from .types import NjdObject
from .sbv2.normalizer import normalize_text
from typing import Union
from pathlib import Path
import kabosu_core
import jpreprocess

def load_marine_model(model_dir: Union[str, None] = None, dict_dir: Union[str, None] = None):
    kabosu_core.load_marine_model(model_dir=model_dir, dict_dir=dict_dir)

def update_global_jtalk_with_user_dict(
        user_dictionary: str | Path | None = None
        ) -> None:
    """Update global openjtalk instance with the user dictionary

    Note that this will change the global state of the openjtalk module.

    """
    kabosu_core.update_global_jtalk_with_user_dict(user_dictionary=user_dictionary)

def extract_fullcontext(
        text: str,
        hankaku: bool = True,
        itaiji: bool = True,
        yomikata: bool = True,
        kanalizer: bool = True,
        use_vanilla: bool = False,
        run_marine: bool = False,
        jpreprocess: Union[jpreprocess.JPreprocess, None] = None
        ) -> list[str]:
    
    """
    ### input 
    text (str): input text  
    ## output
    => list[str] : fullcontext label
    """

    return kabosu_core.extract_fullcontext(
        text=text,
        hankaku=hankaku,
        itaiji=itaiji,
        yomikata=yomikata,
        kanalizer=kanalizer,
        use_vanilla=use_vanilla,
        run_marine=run_marine,
        jpreprocess=jpreprocess
        )


def g2p(
        text: str,
        hankaku: bool = True,
        itaiji: bool = True,
        yomikata: bool = True,
        kanalizer: bool = True,
        use_vanilla: bool = False,
        run_marine: bool = False,
        kana: bool = False,
        join: bool = True,
        jpreprocess: Union[jpreprocess.JPreprocess, None] = None
    ):

    return kabosu_core.g2p(
        text=text,
        hankaku=hankaku,
        itaiji=itaiji,
        yomikata=yomikata,
        kanalizer=kanalizer,
        use_vanilla=use_vanilla,
        run_marine=run_marine,
        kana=kana,
        join=join,
        jpreprocess=jpreprocess
    )

def run_frontend(
            text: str,
            hankaku: bool = True,
            itaiji: bool = True,
            kanalizer: bool = True,
            yomikata: bool = True,
            use_vanilla: bool = False,
            run_marine: bool = False,
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

    return kabosu_core.run_frontend(
            text=text,
            hankaku=hankaku,
            itaiji=itaiji,
            kanalizer=kanalizer,
            yomikata=yomikata,
            use_vanilla=use_vanilla,
            run_marine=run_marine,
            jpreprocess=jpreprocess
            )

def make_label(
        njd_features: list[NjdObject], 
        jpreprocess: Union[jpreprocess.JPreprocess, None] = None
        ) -> list[str]:
    
    return kabosu_core.make_label(
        njd_features=njd_features, 
        jpreprocess=jpreprocess
        )


def reader_furigana(text:str):
    # this liblary use fork version yomikata
    # https://github.com/q9uri/yomikata

    return kabosu_core.reader_furigana(text=text)

def dictreader_furigana(text:str):
    return kabosu_core.dictreader_furigana(text=text)


def kanalizer_convert(text: str):
    return kabosu_core.kanalizer_convert(text=text)

def normalizer(
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
        text = normalize_text(text)

    return  normalizer(
        text=text,
        hankaku=hankaku,
        itaiji=itaiji,
        kanalizer=kanalizer,
        yomikata=yomikata
    ) 


