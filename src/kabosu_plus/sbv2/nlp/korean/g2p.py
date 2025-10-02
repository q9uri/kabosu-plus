import re
from jamo import h2j, j2hcj
from kabosu_core.g2pk4 import G2p
import mecab_ko as MeCab

from kabosu_plus.sbv2.nlp.symbols import PUNCTUATIONS
from kabosu_plus.sbv2.nlp.symbols_ko import HANGUL_CONVERT_LIST, KO_SYMBOLS 
from kabosu_plus.sbv2.nlp import YomiError
from kabosu_plus.sbv2.logging import logger

_g2p = G2p()
_KO_PHONES = KO_SYMBOLS + PUNCTUATIONS
def replace_unknown_mora(phones: list[str], raise_yomi_error: bool = False) -> list[str]:
    new_phones = []
    for phone in phones:
        if phone not in _KO_PHONES:
            if raise_yomi_error:
                raise YomiError(f"unknown phone: {phone}")
            else:
                logger.warning(
                    f'Cannot read unknown {phone}, replaced with "UNK"'
                )
                phone = "UNK"

        new_phones.append(phone)

    return new_phones

tagger = MeCab.Tagger("-Owakati")

rep_map = {
    "/": ",",
    "：": ",",
    ":": ",",
    "；": ",",
    ";": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "．": ".",
    "…": "...",
    "···": "...",
    "・・・": "...",
    "·": ",",
    "・": ",",
    "、": ",",
    "$": ".",
    "“": "'",
    "”": "'",
    '"': "'",
    "‘": "'",
    "’": "'",
    "（": "'",
    "）": "'",
    "(": "'",
    ")": "'",
    "《": "'",
    "》": "'",
    "【": "'",
    "】": "'",
    "[": "'",
    "]": "'",
    "—": "-",
    "−": "-",
    "～": "-",
    "~": "-",
    "「": "'",
    "」": "'",
}


_latin_to_hangul = [
    ('a', '에이'),
    ('b', '비'),
    ('c', '시'),
    ('d', '디'),
    ('e', '이'),
    ('f', '에프'),
    ('g', '지'),
    ('h', '에이치'),
    ('i', '아이'),
    ('j', '제이'),
    ('k', '케이'),
    ('l', '엘'),
    ('m', '엠'),
    ('n', '엔'),
    ('o', '오'),
    ('p', '피'),
    ('q', '큐'),
    ('r', '아르'),
    ('s', '에스'),
    ('t', '티'),
    ('u', '유'),
    ('v', '브이'),
    ('w', '더블유'),
    ('x', '엑스'),
    ('y', '와이'),
    ('z', '제트')
]


def replace_punctuation(text):

    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))
    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)

    # replaced_text = re.sub(
    #     r"[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF\u3005"
    #     + "".join(punctuation)
    #     + r"]+",
    #     "",
    #     replaced_text,
    # )

    return replaced_text

import re

def hangul_number(num, sino=True):
    '''Reference https://github.com/Kyubyong/g2pK'''
    num = re.sub(',', '', num)

    if num == '0':
        return '영'
    if not sino and num == '20':
        return '스무'

    digits = '123456789'
    names = '일이삼사오육칠팔구'
    digit2name = {d: n for d, n in zip(digits, names)}

    modifiers = '한 두 세 네 다섯 여섯 일곱 여덟 아홉'
    decimals = '열 스물 서른 마흔 쉰 예순 일흔 여든 아흔'
    digit2mod = {d: mod for d, mod in zip(digits, modifiers.split())}
    digit2dec = {d: dec for d, dec in zip(digits, decimals.split())}

    spelledout = []
    for i, digit in enumerate(num):
        i = len(num) - i - 1
        if sino:
            if i == 0:
                name = digit2name.get(digit, '')
            elif i == 1:
                name = digit2name.get(digit, '') + '십'
                name = name.replace('일십', '십')
        else:
            if i == 0:
                name = digit2mod.get(digit, '')
            elif i == 1:
                name = digit2dec.get(digit, '')
        if digit == '0':
            if i % 4 == 0:
                last_three = spelledout[-min(3, len(spelledout)):]
                if ''.join(last_three) == '':
                    spelledout.append('')
                    continue
            else:
                spelledout.append('')
                continue
        if i == 2:
            name = digit2name.get(digit, '') + '백'
            name = name.replace('일백', '백')
        elif i == 3:
            name = digit2name.get(digit, '') + '천'
            name = name.replace('일천', '천')
        elif i == 4:
            name = digit2name.get(digit, '') + '만'
            name = name.replace('일만', '만')
        elif i == 5:
            name = digit2name.get(digit, '') + '십'
            name = name.replace('일십', '십')
        elif i == 6:
            name = digit2name.get(digit, '') + '백'
            name = name.replace('일백', '백')
        elif i == 7:
            name = digit2name.get(digit, '') + '천'
            name = name.replace('일천', '천')
        elif i == 8:
            name = digit2name.get(digit, '') + '억'
        elif i == 9:
            name = digit2name.get(digit, '') + '십'
        elif i == 10:
            name = digit2name.get(digit, '') + '백'
        elif i == 11:
            name = digit2name.get(digit, '') + '천'
        elif i == 12:
            name = digit2name.get(digit, '') + '조'
        elif i == 13:
            name = digit2name.get(digit, '') + '십'
        elif i == 14:
            name = digit2name.get(digit, '') + '백'
        elif i == 15:
            name = digit2name.get(digit, '') + '천'
        spelledout.append(name)
    return ''.join(elem for elem in spelledout)


_korean_classifiers = '군데 권 개 그루 닢 대 두 마리 모 모금 뭇 발 발짝 방 번 벌 보루 살 수 술 시 쌈 움큼 정 짝 채 척 첩 축 켤레 톨 통'

def number_to_hangul(text):
    '''Reference https://github.com/Kyubyong/g2pK'''
    tokens = set(re.findall(r'(\d[\d,]*)([\uac00-\ud71f]+)', text))
    for token in tokens:
        num, classifier = token
        if (text[text.find(num)-2] == "쩜"):
            text = text.replace(num, f"{num} ")
            continue
        if classifier[:2] in _korean_classifiers or classifier[0] in _korean_classifiers:
            spelledout = hangul_number(num, sino=False)
        else:
            spelledout = hangul_number(num, sino=True)
        text = text.replace(f'{num}{classifier}', f'{spelledout} {classifier}')
    # digit by digit for remaining digits
    digits = '0123456789'
    names = '영일이삼사오육칠팔구'
    for d, n in zip(digits, names):
        text = text.replace(d, n)
    return text


def normalize_numbers(text):
    # text = re.sub(re.compile(r"([0-9][0-9\,]+[0-9])"), lambda m: m.group(1).replace(",", ""), text)
    text = re.sub(re.compile(r"([0-9]+\.[0-9]+)"), lambda m: m.group(0).replace(".", "쩜 "), text)
    text = re.sub(re.compile(r"\d{1,}kg[^a-zA-Z]"), lambda m: m.group(0).replace("kg", "킬로그램"), text)
    text = re.sub(re.compile(r"\d{1,}g[^a-zA-Z]"), lambda m: m.group(0).replace("g", "그램"), text)
    text = re.sub(re.compile(r"\d{1,}mg[^a-zA-Z]"), lambda m: m.group(0).replace("mg", "밀리그램"), text)
    text = re.sub(re.compile(r"\d{1,}km[^a-zA-Z]"), lambda m: m.group(0).replace("km", "킬로미터"), text)
    text = re.sub(re.compile(r"\d{1,}m[^a-zA-Z]"), lambda m: m.group(0).replace("m", "미터"), text)
    text = re.sub(re.compile(r"\d{1,}cm[^a-zA-Z]"), lambda m: m.group(0).replace("cm", "센티미터"), text)
    text = re.sub(re.compile(r"\d{1,}mm[^a-zA-Z]"), lambda m: m.group(0).replace("mm", "밀리미터"), text)
    text = re.sub(re.compile(r"\d{1,}l[^a-zA-Z]"), lambda m: m.group(0).replace("l", "리터"), text)
    text = re.sub(re.compile(r"\d{1,}ml[^a-zA-Z]"), lambda m: m.group(0).replace("ml", "밀리리터"), text)
    text = re.sub(re.compile(r"\d{1,}bit[^a-zA-Z]"), lambda m: m.group(0).replace("bit", "비트"), text)
    text = re.sub(re.compile(r"\d{1,}B[^a-zA-Z]"), lambda m: m.group(0).replace("B", "바이트"), text)
    text = re.sub(re.compile(r"\d{1,}KB[^a-zA-Z]"), lambda m: m.group(0).replace("KB", "킬로바이트"), text)
    text = re.sub(re.compile(r"\d{1,}MB[^a-zA-Z]"), lambda m: m.group(0).replace("MB", "메가바이트"), text)
    text = re.sub(re.compile(r"\d{1,}GB[^a-zA-Z]"), lambda m: m.group(0).replace("GB", "기가바이트"), text)
    text = re.sub(re.compile(r"\d{1,}TB[^a-zA-Z]"), lambda m: m.group(0).replace("TB", "테라바이트"), text)
    return text

def latin_to_hangul(text):
    for regex, replacement in _latin_to_hangul:
        text = text.replace(regex, replacement)
    return text


def divide_hangul(text):
    text = j2hcj(h2j(text))
    for regex, replacement in HANGUL_CONVERT_LIST:
        text = text.replace(regex, replacement)
    return text


def normalize_text(text):
    text = replace_punctuation(text)
    text = normalize_numbers(text)
    text = number_to_hangul(text)
    text = latin_to_hangul(text)
    text = re.sub(r"([,;.\?\!])([\w])", r"\1 \2", text)
    return text


def distribute_phone(n_phone, n_word):
    phones_per_word = [0] * n_word
    for task in range(n_phone):
        min_tasks = min(phones_per_word)
        min_index = phones_per_word.index(min_tasks)
        phones_per_word[min_index] += 1
    return phones_per_word


def sep_text(text):
    words = re.split(r"([,;.\?\!\s+])", text)
    words = [word for word in words if word.strip() != ""]
    return words


def text_to_words(text):
    tokens = tagger.parse(text).split()
    words = []
    word_lens = []
    for idx, t in enumerate(tokens):
        if (not t.startswith("#")):
            words.append(t)
            word_lens.append(1)
        else:
            words[-1] += t[2:]
            word_lens[-1] += 1
    return words, word_lens

def replace_unk(words, text):
    if "[UNK]" in words:
        print(f"Cannot tokenize \"{text}\". Try to replace [UNK]...")
        for i in range(words.count("[UNK]")):
            loc = words.index("[UNK]")
            if (len(words) == 1):
                words[loc] = text
                continue
            if (loc == 0):
                temp = text[:text.find(words[loc+1])].strip()
                if (temp[-1] in PUNCTUATIONS and words[loc+1] in PUNCTUATIONS):
                    words[loc] = temp[:-1]
                else:
                    words[loc] = temp
                continue
            elif(loc == len(words)-1):
                words[loc] = text[text.find(words[loc-1])+len(words[loc-1]):].strip()
                continue

            if (words[loc+1] != "[UNK]" and words[loc-1] != "[UNK]"):
                temp = text[text.find(words[loc-1])+len(words[loc-1]):text.find(words[loc-1])+text[text.find(words[loc-1]):].find(words[loc+1])].strip()
                if (temp[-1] in PUNCTUATIONS and words[loc+1] in PUNCTUATIONS):
                    words[loc] = temp[:-1]
                else:
                    words[loc] = temp
                continue
            else:
                continue
        
        for i in range(words.count("[UNK]")):
            words.remove("[UNK]")

        return words
    else:
        return words

def g2p(text: str, raise_yomi_error: bool = False):
    norm_text = normalize_text(text)
    phones = []
    tones = []
    phone_len = []
    words, word_lens = text_to_words(norm_text)
    # print(words)
    words = replace_unk(words, norm_text)
    # print(words)
    for word in words:
        if word in PUNCTUATIONS:
            phones.append(word)
            tones.append(0)
            phone_len.append(1)
        else:
            temp_phones = divide_hangul(_g2p(word))
            phones += temp_phones
            tones += [0]*len(temp_phones)
            phone_len.append(len(temp_phones))

    replace_unknown_mora(phones=phones, raise_yomi_error=raise_yomi_error)

    word2ph = []
    for wl, pl in zip(word_lens, phone_len):
        aaa = distribute_phone(pl, wl)
        word2ph += aaa

    phones = ["_"] + phones + ["_"]
    tones = [0] + tones + [0]
    word2ph = [1] + word2ph + [1]
    assert len(phones) == len(tones), norm_text
    assert len(phones) == sum(word2ph), norm_text

    return norm_text, phones, tones, word2ph


if __name__ == "__main__":
    # print(get_dict())
    # print(eng_word_to_phoneme("hello"))

    # print(g2p("In this paper, we propose 1 DSPGAN, a GAN-based universal vocoder."))


    print(g2p("그는 미인 대회 도전이라는 새로운 꿈을 품게 됐고 학교의 허락을 받아내 대회에 출전 ‘미스 콜로라도’로 뽑혔다."))
    
    
    # all_phones = set()
    # for k, syllables in eng_dict.items():
    #     for group in syllables:
    #         for ph in group:
    #             all_phones.add(ph)
    # print(all_phones)
