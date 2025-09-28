from __future__ import annotations

from collections.abc import Sequence
from typing import  Any, Optional, Union

import numpy as np
from numpy.typing import NDArray

from kabosu_plus.sbv2.constants import Languages
from kabosu_plus.sbv2.nlp import llamacpp_embedding_models
from kabosu_plus.sbv2.utils import get_onnx_device_options
from kabosu_plus.sbv2.nlp.japanese.g2p import text_to_sep_kata




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

    # 各単語が何文字かを作る `word2ph` を使う必要があるので、読めない文字は必ず無視する
    # でないと `word2ph` の結果とテキストの文字数結果が整合性が取れない
    text = "".join(text_to_sep_kata(text, raise_yomi_error=False)[0])
    if assist_text:
        assist_text = "".join(text_to_sep_kata(assist_text, raise_yomi_error=False)[0])

    # モデルの読み込み
    model = llamacpp_embedding_models.load_model(
        language=Languages.MULTI,
    )

    style_res_mean = None
    if assist_text:
        # 入力をテンソルに変換
        
        embed_list = model.embed(assist_text)
        style_res = np.array(embed_list, dtype=np.float32)

    embed_list = model.embed(text)
    res = np.array(embed_list, dtype=np.float32)

    zero_array = np.zeros((1, 1024), dtype=np.float32)


    assert len(word2ph) == len(text) + 2, text
    word2phone = word2ph
    phone_level_feature = []
    for i in range(len(word2phone)):
        #先頭と終端のみ埋め込みを入れる
        if i in (0, len(word2phone)-1 ):

            if assist_text:
                assert style_res is not None
                repeat_feature = (
                    np.tile(res, (word2phone[i], 1)) * (1 - assist_text_weight)
                    + np.tile(style_res, (word2phone[i], 1)) * assist_text_weight
                )
            else:
                #先頭と終端のみ埋め込みを入れる
                repeat_feature = np.tile(res, (word2phone[i], 1))

        else:
            #先頭と終端以外は0を入れる
            repeat_feature = np.tile(zero_array, (word2phone[i], 1))

        phone_level_feature.append(repeat_feature)

    phone_level_feature = np.concatenate(phone_level_feature, axis=0)

    return phone_level_feature.T
