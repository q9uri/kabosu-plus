<!--licence-->
<a href="./LICENSE">
    <img src="https://img.shields.io/badge/-AGPL3 Licence-5674bc.svg?">
</a>
<br>
<br>
<!--logo-->>
<p align="center">
<img width="70"  src = "./kabosu_icon.png" />
</p>

## kabosu-plus ~Japanese tokenizer for TTS~


### 使用許諾 (義務ではない)
著作権侵害、ディープフェイク、その他の犯罪のために使用しないでください
許諾を得た音声もしくは自分の音声にのみ使用してください

### 企業の皆様へ
このソフトウェアを使用したい場合は事前にお問い合わせください。

---
requirements
jpreprocess dictonary(auto download)  
一回目は辞書をダウンロードしますが、見つからないので落ちます。
yomikata bert model

```
# download bert model  
python -m yomikata download

```
https://github.com/passaglia/yomikata  
https://github.com/VOICEVOX/kanalizer  
https://github.com/ikegami-yukino/jaconv