"""
Style-Bert-VITS2 の ONNX 推論に必要な各言語ごとの ONNX 版 BERT モデルをロード/取得するためのモジュール。
このモジュールは style_bert_vits2.nlp.bert_models での実装を ONNX 推論向けに変更したもの。

オリジナルの Bert-VITS2 では各言語ごとの BERT モデルが初回インポート時にハードコードされたパスから「暗黙的に」ロードされているが、
場合によっては多重にロードされて非効率なほか、BERT モデルのロード元のパスがハードコードされているためライブラリ化ができない。

そこで、ライブラリの利用前に、音声合成に利用する言語の BERT モデルだけを「明示的に」ロードできるようにした。
一度 load_model/tokenizer() で当該言語の BERT モデルがロードされていれば、ライブラリ内部のどこからでもロード済みのモデル/トークナイザーを取得できる。
"""

import gc
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional, Union


from huggingface_hub import hf_hub_download
from llama_cpp import Llama

from kabosu_plus.sbv2.constants import Languages, DEFAULT_ONNX_BERT_MODEL_PATHS
from kabosu_plus.sbv2.logging import logger



# 各言語ごとのロード済みの BERT モデルを格納する辞書
__loaded_models: dict[Languages, Any] = {}


def load_model(
    language: Languages,
    pretrained_model_name_or_path: Optional[str] = None,
    onnx_providers: Sequence[Union[str, tuple[str, dict[str, Any]]]] = [("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"})],
    cache_dir: Optional[str] = None,
    revision: str = "main",
    enable_cpu_mem_arena: bool | None = None,
) -> onnxruntime.InferenceSession:  # fmt: skip
    """
    指定された言語の ONNX 版 BERT モデルをロードし、ロード済みの ONNX 版 BERT モデルを返す。
    一度ロードされていれば、ロード済みの ONNX 版 BERT モデルを即座に返す。
    ライブラリ利用時は常に必ず pretrain_model_name_or_path (Hugging Face のリポジトリ名 or ローカルのファイルパス) を指定する必要がある。
    ロードにはそれなりに時間がかかるため、ライブラリ利用前に明示的に pretrained_model_name_or_path を指定してロードしておくべき。
    cache_dir と revision は pretrain_model_name_or_path がリポジトリ名の場合のみ有効。

    Style-Bert-VITS2 では、ONNX 版 BERT モデルに下記の 3 つが利用されている。
    これ以外の ONNX 版 BERT モデルを指定した場合は正常に動作しない可能性が高い。
    - 日本語: tsukumijima/deberta-v2-large-japanese-char-wwm-onnx

    Args:
        language (Languages): ロードする学習済みモデルの対象言語
        pretrained_model_name_or_path (Optional[str]): ロードする学習済みモデルの名前またはパス。指定しない場合はデフォルトのパスが利用される (デフォルト: None)
        onnx_providers (list[str]): ONNX 推論で利用する ExecutionProvider (CPUExecutionProvider, CUDAExecutionProvider など)
        cache_dir (Optional[str]): モデルのキャッシュディレクトリ。指定しない場合はデフォルトのキャッシュディレクトリが利用される (デフォルト: None)
        revision (str): モデルの Hugging Face 上の Git リビジョン。指定しない場合は最新の main ブランチの内容が利用される (デフォルト: None)
        enable_cpu_mem_arena (bool | None): CPU 推論時にもメモリアリーナを有効化するかどうか。デフォルトでは GPU 推論時のみ有効化される (デフォルト: None)

    Returns:
        onnxruntime.InferenceSession: ロード済みの BERT モデル
    """

    # すでにロード済みの場合はそのまま返す
    if language in __loaded_models:
        return __loaded_models[language]

    # pretrained_model_name_or_path が指定されていない場合はデフォルトのパスを利用
    if pretrained_model_name_or_path is None:
        assert DEFAULT_ONNX_BERT_MODEL_PATHS[language].exists(), \
            f"The default {language.name} BERT tokenizer does not exist on the file system. Please specify the path to the pre-trained model."  # fmt: skip
        pretrained_model_name_or_path = str(DEFAULT_ONNX_BERT_MODEL_PATHS[language])


    # pretrained_model_name_or_path に Hugging Face のリポジトリ名が指定された場合 (aaaa/bbbb のフォーマットを想定):
    # 指定された revision の ONNX 版 BERT モデルを cache_dir にダウンロードする (既にダウンロード済みの場合は何も行われない)
    if len(pretrained_model_name_or_path.split("/")) == 2:
        model_path = Path(
            hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                filename="model_fp16.onnx",
                cache_dir=cache_dir,
                revision=revision,
            )
        )

    # pretrained_model_name_or_path にファイルパスが指定された場合:
    # 既にダウンロード済みという前提のもと、モデルへのローカルパスを model_path に格納する
    else:
        model_path = Path(pretrained_model_name_or_path).resolve() / "model_fp16.onnx"


    # BERT モデルをロードし、辞書に格納して返す
    start_time = time.time()
    __loaded_models[language] = Llama(
        model_path=model_path,
        embedding=True,
        flash_attn=True,
    )
    logger.info(
        f"Loaded the {language.name} ONNX BERT model from {pretrained_model_name_or_path} ({time.time() - start_time:.2f}s)"
    )

    return __loaded_models[language]



def is_model_loaded(language: Languages) -> bool:
    """
    指定された言語の ONNX 版 BERT モデルがロード済みかどうかを返す。
    """

    return language in __loaded_models



def unload_model(language: Languages) -> None:
    """
    指定された言語の ONNX 版 BERT モデルをアンロードする。

    Args:
        language (Languages): アンロードする BERT モデルの言語
    """

    if language in __loaded_models:
        del __loaded_models[language]
        gc.collect()
        logger.info(f"Unloaded the {language.name} ONNX BERT model")

def unload_all_models() -> None:
    """
    すべての ONNX 版 BERT モデルをアンロードする。
    """

    for language in list(__loaded_models.keys()):
        unload_model(language)
    logger.info("Unloaded all ONNX BERT models")
