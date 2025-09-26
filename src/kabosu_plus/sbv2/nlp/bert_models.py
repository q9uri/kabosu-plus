"""
Style-Bert-VITS2 の学習・推論に必要な各言語ごとの BERT モデルをロード/取得するためのモジュール。

オリジナルの Bert-VITS2 では各言語ごとの BERT モデルが初回インポート時にハードコードされたパスから「暗黙的に」ロードされているが、
場合によっては多重にロードされて非効率なほか、BERT モデルのロード元のパスがハードコードされているためライブラリ化ができない。

そこで、ライブラリの利用前に、音声合成に利用する言語の BERT モデルだけを「明示的に」ロードできるようにした。
一度 load_model/tokenizer() で当該言語の BERT モデルがロードされていれば、ライブラリ内部のどこからでもロード済みのモデル/トークナイザーを取得できる。
"""

from __future__ import annotations

import gc
from typing import Optional, Union, cast

from transformers import (
    DebertaV2TokenizerFast,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)


from kabosu_plus.sbv2.constants import Languages
from kabosu_plus.sbv2.logging import logger
from kabosu_plus.sbv2.nlp import onnx_bert_models




# 各言語ごとのロード済みの BERT トークナイザーを格納する辞書
__loaded_tokenizers: dict[
    Languages,
    Union[PreTrainedTokenizer, PreTrainedTokenizerFast, DebertaV2TokenizerFast],
] = {}



def load_tokenizer(
    language: Languages,
    pretrained_model_name_or_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    revision: str = "main",
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast, DebertaV2TokenizerFast]:
    """
    指定された言語の BERT トークナイザーをロードし、ロード済みの BERT トークナイザーを返す。
    一度ロードされていれば、ロード済みの BERT トークナイザーを即座に返す。
    ライブラリ利用時は常に必ず pretrain_model_name_or_path (Hugging Face のリポジトリ名 or ローカルのファイルパス) を指定する必要がある。
    ロードにはそれなりに時間がかかるため、ライブラリ利用前に明示的に pretrained_model_name_or_path を指定してロードしておくべき。
    cache_dir と revision は pretrain_model_name_or_path がリポジトリ名の場合のみ有効。

    Style-Bert-VITS2 では、BERT モデルに下記の 3 つが利用されている。
    これ以外の BERT モデルを指定した場合は正常に動作しない可能性が高い。
    - 日本語: ku-nlp/deberta-v2-large-japanese-char-wwm
    - 英語: microsoft/deberta-v3-large
    - 中国語: hfl/chinese-roberta-wwm-ext-large

    Args:
        language (Languages): ロードする学習済みモデルの対象言語
        pretrained_model_name_or_path (Optional[str]): ロードする学習済みモデルの名前またはパス。指定しない場合はデフォルトのパスが利用される (デフォルト: None)
        cache_dir (Optional[str]): モデルのキャッシュディレクトリ。指定しない場合はデフォルトのキャッシュディレクトリが利用される (デフォルト: None)
        revision (str): モデルの Hugging Face 上の Git リビジョン。指定しない場合は最新の main ブランチの内容が利用される (デフォルト: None)

    Returns:
        Union[PreTrainedTokenizer, PreTrainedTokenizerFast, DebertaV2TokenizerFast]: ロード済みの BERT トークナイザー
    """

    # すでにロード済みの場合はそのまま返す
    if language in __loaded_tokenizers:
        return __loaded_tokenizers[language]

    # pretrained_model_name_or_path が指定されていない場合はデフォルトのパスを利用
    if pretrained_model_name_or_path is None:
        # ライブラリ利用時、特例的にこの状況で ONNX 版 BERT トークナイザーがロードされている場合はそのまま返す
        ## ONNX 版 BERT トークナイザー単独で g2p 処理を行うために必要 (各言語の g2p.py はこの関数に依存している)
        ## 設計的には微妙だがこの方が差異を吸収できて手っ取り早い
        if onnx_bert_models.is_tokenizer_loaded(language):  # fmt: skip
            return onnx_bert_models.load_tokenizer(language)

    return __loaded_tokenizers[language]


def is_tokenizer_loaded(language: Languages) -> bool:
    """
    指定された言語の BERT トークナイザーがロード済みかどうかを返す。
    """

    return language in __loaded_tokenizers


def unload_tokenizer(language: Languages) -> None:
    """
    指定された言語の BERT トークナイザーをアンロードする。

    Args:
        language (Languages): アンロードする BERT トークナイザーの言語
    """

    if language in __loaded_tokenizers:
        del __loaded_tokenizers[language]
        gc.collect()
        logger.info(f"Unloaded the {language.name} BERT tokenizer")



def unload_all_tokenizers() -> None:
    """
    すべての BERT トークナイザーをアンロードする。
    """

    for language in list(__loaded_tokenizers.keys()):
        unload_tokenizer(language)
    logger.info("Unloaded all BERT tokenizers")
