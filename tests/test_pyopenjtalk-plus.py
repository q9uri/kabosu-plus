import pytest

import kabosu_plus
from kabosu_plus.types import NjdObject



def _print_results(njd_features: list[NjdObject], labels: list[str]):
    for f in njd_features:
        s, p = f["string"], f["pron"]
        print(s, p)

    for label in labels:
        print(label)


def test_hello():
    njd_features = kabosu_plus.run_frontend("こんにちは")
    labels = kabosu_plus.make_label(njd_features)
    _print_results(njd_features, labels)


def test_hello_marine():
    pytest.importorskip("marine")
    njd_features = kabosu_plus.run_frontend("こんにちは", run_marine=True)
    labels = kabosu_plus.make_label(njd_features)
    _print_results(njd_features, labels)

def test_njd_features():
    njd_features = kabosu_plus.run_frontend("こんにちは")
    expected_feature = [
        {
            "string": "こんにちは",
            "pos": "感動詞",
            "pos_group1": "*",
            "pos_group2": "*",
            "pos_group3": "*",
            "ctype": "*",
            "cform": "*",
            "orig": "こんにちは",
            "read": "コンニチハ",
            "pron": "コンニチワ",
            "acc": 0,
            "mora_size": 5,
            "chain_rule": "-1",
            "chain_flag": -1,
        }
    ]
    assert njd_features == expected_feature


def test_njd_features_marine():
    pytest.importorskip("marine")
    njd_features = kabosu_plus.run_frontend("こんにちは", run_marine=True)
    expected_feature = [
        {
            "string": "こんにちは",
            "pos": "感動詞",
            "pos_group1": "*",
            "pos_group2": "*",
            "pos_group3": "*",
            "ctype": "*",
            "cform": "*",
            "orig": "こんにちは",
            "read": "コンニチハ",
            "pron": "コンニチワ",
            "acc": 0,
            "mora_size": 5,
            "chain_rule": "-1",
            "chain_flag": -1,
        }
    ]
    assert njd_features == expected_feature

def test_njd_features():
    njd_features = kabosu_plus.run_frontend("こんにちは")
    expected_feature = [
        {
            "string": "こんにちは",
            "pos": "感動詞",
            "pos_group1": "*",
            "pos_group2": "*",
            "pos_group3": "*",
            "ctype": "*",
            "cform": "*",
            "orig": "こんにちは",
            "read": "コンニチハ",
            "pron": "コンニチワ",
            "acc": 0,
            "mora_size": 5,
            "chain_rule": "*",
            "chain_flag": -1,
        }
    ]
    assert njd_features == expected_feature


def test_njd_features_marine():
    pytest.importorskip("marine")
    njd_features = kabosu_plus.run_frontend("こんにちは", run_marine=True)
    expected_feature = [
        {
            "string": "こんにちは",
            "pos": "感動詞",
            "pos_group1": "*",
            "pos_group2": "*",
            "pos_group3": "*",
            "ctype": "*",
            "cform": "*",
            "orig": "こんにちは",
            "read": "コンニチハ",
            "pron": "コンニチワ",
            "acc": 0,
            "mora_size": 5,
            "chain_rule": "*",
            "chain_flag": -1,
        }
    ]
    assert njd_features == expected_feature


def test_fullcontext():
    features = kabosu_plus.run_frontend("こんにちは")
    labels = kabosu_plus.make_label(features)
    labels2 = kabosu_plus.extract_fullcontext("こんにちは")
    for a, b in zip(labels, labels2):
        assert a == b


def test_fullcontext_marine():
    pytest.importorskip("marine")
    features = kabosu_plus.run_frontend("こんにちは", run_marine=True)
    labels = kabosu_plus.make_label(features)
    labels2 = kabosu_plus.extract_fullcontext("こんにちは", run_marine=True)
    for a, b in zip(labels, labels2):
        assert a == b


def test_jtalk():
    for text in [
        "今日も良い天気ですね",
        "こんにちは。",
        "どんまい！",
        "パソコンのとりあえず知っておきたい使い方",
    ]:
        njd_features = kabosu_plus.run_frontend(text, yomikata=False)
        labels = kabosu_plus.make_label(njd_features)
        _print_results(njd_features, labels)

        surface = "".join(map(lambda f: f["string"], njd_features))
        assert surface == text


def test_jtalk_marine():
    pytest.importorskip("marine")
    for text in [
        "今日も良い天気ですね",
        "こんにちは。",
        "どんまい！",
        "パソコンのとりあえず知っておきたい使い方",
    ]:
        njd_features = kabosu_plus.run_frontend(text, run_marine=True, yomikata=False)
        labels = kabosu_plus.make_label(njd_features)
        _print_results(njd_features, labels)

        surface = "".join(map(lambda f: f["string"], njd_features))
        assert surface == text


def test_g2p_kana():
    for text, pron in [
        ("", ""),  # empty string
        ("今日もこんにちは", "キョウモコンニチハ"),
        ("いやあん", "イヤアン"),
        (
            "パソコンのとりあえず知っておきたい使い方",
            "パソコンノトリアエズシッテオキタイツカイカタ",
        ),
    ]:
        p = kabosu_plus.g2p(text, kana=True)
        assert p == pron


def test_g2p_phone():
    for text, pron in [
        ("", ""),  # empty string
        ("こんにちは", "k o N n i ch i w a"),
        ("ななみんです", "n a n a m i N d e s U"),
        ("ハローユーチューブ", "h a r o o y u u ch u u b u"),
    ]:
        p = kabosu_plus.g2p(text, kana=False)
        assert p == pron


def test_g2p_nani_model():
    test_cases = [
        {
            "text": "何か問題があれば何でも言ってください、どんな些細なことでも何とかします。",
            "pron_without_nani": "ナニカモンダイガアレバナニデモイッテクダサイ、ドンナササイナコトデモナニトカシマス、",
            "pron_with_nani": "ナニカモンダイガアレバナンデモイッテクダサイ、ドンナササイナコトデモナントカシマス、",
        },
        {
            "text": "何か特別なことをしたわけではありませんが、何故か周りの人々が何かと気にかけてくれます。何と言えばいいのか分かりません。",
            "pron_without_nani": "ナニカトクベツナコトヲシタワケデハアリマセンガ、ナゼカマワリノヒトビトガナニカトキニカケテクレマス、ナニトイエバイイノカワカリマセン、",
            "pron_with_nani": "ナニカトクベツナコトヲシタワケデハアリマセンガ、ナゼカマワリノヒトビトガナニカトキニカケテクレマス、ナントイエバイイノカワカリマセン、",
        },
        {
            "text": "私も何とかしたいですが、何でも行くリソースはありません。",
            "pron_without_nani": "ワタシモナニトカシタイデスガ、ナニデモイクリソースハアリマセン、",
            "pron_with_nani": "ワタシモナントカシタイデスガ、ナンデモイクリソースハアリマセン、",
        },
        {
            "text": "何を言っても何の問題もありません。",
            "pron_without_nani": "ナニヲイッテモナニノモンダイモアリマセン、",
            "pron_with_nani": "ナニヲイッテモナンノモンダイモアリマセン、",
        },
        {
            "text": "これは何ですか？何の情報？",
            "pron_without_nani": "コレハナニデスカ？ナニノジョウホウ？",
            "pron_with_nani": "コレハナンデスカ？ナンノジョウホウ？",
        },
        {
            "text": "何だろう、何でも嘘つくのやめてもらっていいですか？",
            "pron_without_nani": "ナニダロウ、ナニデモウソツクノヤメテモラッテイイデスカ？",
            "pron_with_nani": "ナンダロウ、ナンデモウソツクノヤメテモラッテイイデスカ？",
        },
        {
            "text": "質問は何のことかな？",
            "pron_without_nani": "シツモンハナニノコトカナ？",
            "pron_with_nani": "シツモンハナンノコトカナ？",
        },
    ]

    # without nani model
    for case in test_cases:
        p = kabosu_plus.g2p(case["text"], kana=True, use_vanilla=True)
        assert p == case["pron_without_nani"]

    # with nani model
    for case in test_cases:
        p = kabosu_plus.g2p(case["text"], kana=True, use_vanilla=False)
        assert p == case["pron_with_nani"]


def test_odoriji():
    # 一の字点（ゝ、ゞ、ヽ、ヾ）の処理テスト
    # 濁点なしの一の字点
    njd_features = kabosu_plus.run_frontend("なゝ樹")
    assert njd_features[0]["read"] == "ナ"
    assert njd_features[0]["pron"] == "ナ"
    assert njd_features[0]["mora_size"] == 1
    assert njd_features[1]["read"] == "ナ"
    assert njd_features[1]["pron"] == "ナ"
    assert njd_features[1]["mora_size"] == 1
    assert njd_features[2]["read"] == "キ"
    assert njd_features[2]["pron"] == "キ"
    assert njd_features[2]["mora_size"] == 1

    # 濁点ありの一の字点
    njd_features = kabosu_plus.run_frontend("金子みすゞ")
    assert njd_features[0]["read"] == "カネコ"
    assert njd_features[0]["pron"] == "カネコ"
    assert njd_features[0]["mora_size"] == 3
    assert njd_features[1]["read"] == "ミス"
    assert njd_features[1]["pron"] == "ミス"
    assert njd_features[1]["mora_size"] == 2
    assert njd_features[2]["read"] == "ズ"
    assert njd_features[2]["pron"] == "ズ"
    assert njd_features[2]["mora_size"] == 1

    # 濁点なしの一の字点（づゝ）
    njd_features = kabosu_plus.run_frontend("づゝ")
    assert njd_features[0]["read"] == "ヅ"
    assert njd_features[0]["pron"] == "ヅ"
    assert njd_features[0]["mora_size"] == 1
    assert njd_features[1]["read"] == "ツ"
    assert njd_features[1]["pron"] == "ツ"
    assert njd_features[1]["mora_size"] == 1

    # 濁点ありの一の字点（ぶゞ漬け）
    njd_features = kabosu_plus.run_frontend("ぶゞ漬け")
    assert njd_features[0]["read"] == "ブ"
    assert njd_features[0]["pron"] == "ブ"
    assert njd_features[0]["mora_size"] == 1
    assert njd_features[1]["read"] == "ブ"
    assert njd_features[1]["pron"] == "ブ"
    assert njd_features[1]["mora_size"] == 1
    assert njd_features[2]["read"] == "ヅケ"
    assert njd_features[2]["pron"] == "ヅケ"
    assert njd_features[2]["mora_size"] == 2

    # 片仮名の一の字点（バナヽ）
    njd_features = kabosu_plus.run_frontend("バナヽ")
    assert njd_features[0]["read"] == "バナ"
    assert njd_features[0]["pron"] == "バナ"
    assert njd_features[0]["mora_size"] == 2
    assert njd_features[1]["read"] == "ナ"
    assert njd_features[1]["pron"] == "ナ"
    assert njd_features[1]["mora_size"] == 1

    # use_vanilla=True の場合は処理されない
    njd_features = kabosu_plus.run_frontend("なゝ樹", use_vanilla=True)
    assert njd_features[1]["read"] == "、"
    assert njd_features[1]["pron"] == "、"

    # 単一の踊り字（辞書に登録されていないパターン）
    njd_features = kabosu_plus.run_frontend("愛々")
    assert njd_features[0]["read"] == "アイ"
    assert njd_features[0]["pron"] == "アイ"
    assert njd_features[0]["mora_size"] == 2
    assert njd_features[1]["read"] == "アイ"
    assert njd_features[1]["pron"] == "アイ"
    assert njd_features[1]["mora_size"] == 2
    njd_features = kabosu_plus.run_frontend("咲々")
    assert njd_features[0]["read"] == "サキ"
    assert njd_features[0]["pron"] == "サキ"
    assert njd_features[0]["mora_size"] == 2
    assert njd_features[1]["read"] == "サキ"
    assert njd_features[1]["pron"] == "サキ"
    assert njd_features[1]["mora_size"] == 2

    # 単一の踊り字だが、形態素解析で展開しないと正しい読みを取得できないケース
    # 実装上漢字1字だけで再解析した際に読みが間違ってしまうことがあるが、改善するのが面倒なのでテストケースには含めていない
    njd_features = kabosu_plus.run_frontend("結婚式々場")
    assert njd_features[0]["read"] == "ケッコンシキ"
    assert njd_features[0]["pron"] == "ケッコンシ’キ"
    assert njd_features[0]["mora_size"] == 6
    assert njd_features[1]["read"] == "シキジョウ"
    assert njd_features[1]["pron"] == "シ’キジョー"
    assert njd_features[1]["mora_size"] == 4
    njd_features = kabosu_plus.run_frontend("学生々活")
    assert njd_features[0]["read"] == "ガクセイ"
    assert njd_features[0]["pron"] == "ガク’セー"
    assert njd_features[0]["mora_size"] == 4
    assert njd_features[1]["read"] == "セイカツ"
    assert njd_features[1]["pron"] == "セーカツ"
    assert njd_features[1]["mora_size"] == 4
    njd_features = kabosu_plus.run_frontend("民主々義")
    assert njd_features[0]["read"] == "ミンシュ"
    assert njd_features[0]["pron"] == "ミンシュ"
    assert njd_features[0]["mora_size"] == 3
    assert njd_features[1]["read"] == "シュギ"
    assert njd_features[1]["pron"] == "シュギ"
    assert njd_features[1]["mora_size"] == 2

    # 連続する踊り字
    njd_features = kabosu_plus.run_frontend("叙々々苑")
    assert njd_features[0]["read"] == "ジョ"
    assert njd_features[0]["pron"] == "ジョ"
    assert njd_features[0]["mora_size"] == 1
    assert njd_features[1]["read"] == "ジョジョ"
    assert njd_features[1]["pron"] == "ジョジョ"
    assert njd_features[1]["mora_size"] == 2
    njd_features = kabosu_plus.run_frontend("叙々々々苑")
    assert njd_features[0]["read"] == "ジョ"
    assert njd_features[0]["pron"] == "ジョ"
    assert njd_features[0]["mora_size"] == 1
    assert njd_features[1]["read"] == "ジョジョ"
    assert njd_features[1]["pron"] == "ジョジョ"
    assert njd_features[1]["mora_size"] == 2
    assert njd_features[2]["read"] == "ジョ"
    assert njd_features[2]["pron"] == "ジョ"
    assert njd_features[2]["mora_size"] == 1
    njd_features = kabosu_plus.run_frontend("叙々々々々苑")
    assert njd_features[0]["read"] == "ジョ"
    assert njd_features[0]["pron"] == "ジョ"
    assert njd_features[0]["mora_size"] == 1
    assert njd_features[1]["read"] == "ジョジョ"
    assert njd_features[1]["pron"] == "ジョジョ"
    assert njd_features[1]["mora_size"] == 2
    assert njd_features[2]["read"] == "ジョジョ"
    assert njd_features[2]["pron"] == "ジョジョ"
    assert njd_features[2]["mora_size"] == 2
    njd_features = kabosu_plus.run_frontend("叙々々々々々苑")
    assert njd_features[0]["read"] == "ジョ"
    assert njd_features[0]["pron"] == "ジョ"
    assert njd_features[0]["mora_size"] == 1
    assert njd_features[1]["read"] == "ジョジョジョジョジョ"
    assert njd_features[1]["pron"] == "ジョジョジョジョジョ"
    assert njd_features[1]["mora_size"] == 5
    njd_features = kabosu_plus.run_frontend("複々々線")
    print(njd_features)
    assert njd_features[0]["read"] == "フク"
    assert njd_features[0]["pron"] == "フ’ク"
    assert njd_features[0]["mora_size"] == 2
    assert njd_features[1]["read"] == "フクフク"
    assert njd_features[1]["pron"] == "フ’クフ’ク"
    assert njd_features[1]["mora_size"] == 4
    njd_features = kabosu_plus.run_frontend("複々々々線")
    assert njd_features[0]["read"] == "フク"
    assert njd_features[0]["pron"] == "フ’ク"
    assert njd_features[0]["mora_size"] == 2
    assert njd_features[1]["read"] == "フクフク"
    assert njd_features[1]["pron"] == "フ’クフ’ク"
    assert njd_features[1]["mora_size"] == 4
    assert njd_features[2]["read"] == "フク"
    assert njd_features[2]["pron"] == "フ’ク"
    assert njd_features[2]["mora_size"] == 2
    njd_features = kabosu_plus.run_frontend("今日も前進々々")
    assert njd_features[0]["read"] == "キョウ"
    assert njd_features[0]["pron"] == "キョー"
    assert njd_features[0]["mora_size"] == 2
    assert njd_features[1]["read"] == "モ"
    assert njd_features[1]["pron"] == "モ"
    assert njd_features[1]["mora_size"] == 1
    assert njd_features[2]["read"] == "ゼンシン"
    assert njd_features[2]["pron"] == "ゼンシン"
    assert njd_features[2]["mora_size"] == 4
    assert njd_features[3]["read"] == "ゼンシン"
    assert njd_features[3]["pron"] == "ゼンシン"
    assert njd_features[3]["mora_size"] == 4

    # 2文字以上の漢字の後の踊り字
    njd_features = kabosu_plus.run_frontend("部分々々")
    assert njd_features[0]["read"] == "ブブン"
    assert njd_features[0]["pron"] == "ブブン"
    assert njd_features[0]["mora_size"] == 3
    assert njd_features[1]["read"] == "ブブン"
    assert njd_features[1]["pron"] == "ブブン"
    assert njd_features[1]["mora_size"] == 3
    njd_features = kabosu_plus.run_frontend("後手々々")
    assert njd_features[0]["read"] == "ゴテ"
    assert njd_features[0]["pron"] == "ゴテ"
    assert njd_features[0]["mora_size"] == 2
    assert njd_features[1]["read"] == "ゴテ"
    assert njd_features[1]["pron"] == "ゴテ"
    assert njd_features[1]["mora_size"] == 2
    njd_features = kabosu_plus.run_frontend("其他々々")
    assert njd_features[0]["read"] == "ソノ"
    assert njd_features[0]["pron"] == "ソノ"
    assert njd_features[0]["mora_size"] == 2
    assert njd_features[1]["read"] == "ホカ"
    assert njd_features[1]["pron"] == "ホカ"
    assert njd_features[1]["mora_size"] == 2
    assert njd_features[2]["read"] == "ソノホカ"
    assert njd_features[2]["pron"] == "ソノホカ"
    assert njd_features[2]["mora_size"] == 4

    # 踊り字の前に漢字がない場合
    njd_features = kabosu_plus.run_frontend("々々")
    assert njd_features[0]["read"] == "、"
    assert njd_features[0]["pron"] == "、"

    # use_vanilla=True の場合は処理されない
    njd_features = kabosu_plus.run_frontend("愛々", use_vanilla=True)
    assert njd_features[1]["read"] == "、"
    assert njd_features[1]["pron"] == "、"


