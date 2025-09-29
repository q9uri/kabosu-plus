from __future__ import annotations

from collections.abc import Sequence
from typing import  Any, Optional, Union

import numpy as np
from numpy.typing import NDArray

from kabosu_plus.sbv2.constants import Languages
from kabosu_plus.sbv2.nlp import llamacpp_embedding_models
from kabosu_plus.sbv2.nlp import onnx_bert_models

from kabosu_plus.sbv2.nlp import language_selector


def extract_bert_feature_lammacpp(
    text: str,
    word2ph: list[int],
    assist_text: Optional[str] = None,
    assist_text_weight: float = 0.7,
) -> NDArray[Any]:
    """
    日本語のテキストから BERT の特徴量を抽出する (ONNX 推論)

    Args:
        text (str): 日本語のテキスト
        word2ph (list[int]): 元のテキストの各文字に音素が何個割り当てられるかを表すリスト
        onnx_providers (list[str]): ONNX 推論で利用する ExecutionProvider (CPUExecutionProvider, CUDAExecutionProvider など)
        assist_text (Optional[str], optional): 補助テキスト (デフォルト: None)
        assist_text_weight (float, optional): 補助テキストの重み (デフォルト: 0.7)

    Returns:
        NDArray[Any]: BERT の特徴量
    """

    # モデルの読み込み
    model = llamacpp_embedding_models.load_model(
        language=Languages.MULTI,
    )

    if assist_text:
        # 入力をテンソルに変換
        
        embed_list = model.embed(assist_text)
        style_res = np.array(embed_list, dtype=np.float32)

    embed_list = model.embed(text)
    res = np.array(embed_list, dtype=np.float32)

    zero_array = np.zeros((1, 1024), dtype=np.float32)

    language_type = language_selector(text, [Languages.JP, Languages.ZH, Languages.EN])
    
    if language_type in ("JP", "ZH"):
        assert len(word2ph) == len(text) + 2, text
    
    elif language_type == "EN":
        tokenizer = onnx_bert_models.load_tokenizer(Languages.EN)
        tokens = tokenizer.tokenize(text)
        assert len(word2ph) == len(tokens) +2 , (text, tokens, len(word2ph), len(tokens))


    phone_level_feature = []
    for i in range(len(word2ph)):
        #先頭と終端のみ埋め込みを入れる
        if i in (0, len(word2ph)-1 ):

            if assist_text:
                assert style_res is not None
                repeat_feature = (
                    np.tile(res, (word2ph[i], 1)) * (1 - assist_text_weight)
                    + np.tile(style_res, (word2ph[i], 1)) * assist_text_weight
                )
            else:
                #先頭と終端のみ埋め込みを入れる
                repeat_feature = np.tile(res, (word2ph[i], 1))

        else:
            #先頭と終端以外は0を入れる
            repeat_feature = np.tile(zero_array, (word2ph[i], 1))

        phone_level_feature.append(repeat_feature)

    phone_level_feature = np.concatenate(phone_level_feature, axis=0)

    return phone_level_feature.T
