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

def test_g2p():
    output = kabosu_plus.g2p("そして、畳の表は、すでに幾年前に換えられたのか分らなかった。", kana=True)
    assert output == "ソシテ、タタミノオモテハ、スデニイクネンマエニカエラレタノカワカラナカッタ、"
    output = kabosu_plus.g2p("you are so cute! mii-chan!", kana=True)
    assert output == "ユー、アー、ソー、キュート、、ミイ、チャン、"
    print(output)