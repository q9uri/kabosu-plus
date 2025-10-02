import pytest
import kabosu_plus


def test_yomikata_reader_furigana_test():
    text = 'そして、畳の表は、すでに幾年前に換えられたのか分らなかった。'
    output = kabosu_plus.reader_furigana(text)
    assert output == "そして、畳の{表/おもて}は、すでに幾年前に換えられたのか分らなかった。"

def test_yomikata_dictreader_furigana_test():
    text = 'そして、畳の表は、すでに幾年前に換えられたのか分らなかった。'
    output = kabosu_plus.dictreader_furigana(text)
    assert output == "そして、{畳/たたみ}の{表/ひょう}は、すでに{幾/いく}{年/ねん}{前/まえ}に{換/か}えられたのか{分/わか}らなかった。"

def test_kanalizer_convert_test():
    output = kabosu_plus.kanalizer_convert("lemon")
    assert output == "レモン"
    print(output)

def test_g2p_yomikata():
    text = kabosu_plus.normalize_text("そして、畳の表は、すでに幾年前に換えられたのか分らなかった。")
    output = kabosu_plus.g2p(text, kana=True)
    assert output == "ソシテ、タタミノオモテハ、スデニイクネンマエニカエラレタノカワカラナカッタ、"

def test_g2p_hungl():
    text = kabosu_plus.normalize_text("이봐, 센파이. 한국으로 여행하자? 현지의 맛있는 요리를 먹으면 좋겠다.")
    output = kabosu_plus.g2p(text, kana=True)
    assert output == "イブァ、ゼンパイ、ハンググロヨヘンハザ？ ホンジウィマジンヌノリルルモグモンゾゲッッタ、"
    print(output)

def test_g2p_kanalizer():
    text = kabosu_plus.normalize_text("you are so cute! mii-chan!")
    output = kabosu_plus.g2p(text, kana=True)
    assert output == "ユーアーソーキュート、ミイチャン、"

def test_njd_features_to_babytalk():

    fullcontext = kabosu_plus.extract_fullcontext("可愛い赤ちゃんですねー。触ってもいいですか？")
    babytalk_fullcontext = kabosu_plus.extract_fullcontext("可愛い赤ちゃんですねー。触ってもいいですか？", babytalk=True)
    assert fullcontext != babytalk_fullcontext

def test_njd_features_to_dakuten():

    fullcontext = kabosu_plus.extract_fullcontext("それでも、僕は知らないッ")
    dakuten_fullcontext = kabosu_plus.extract_fullcontext("それでも、僕は知らないッ",  dakuten=True)
    assert fullcontext != dakuten_fullcontext