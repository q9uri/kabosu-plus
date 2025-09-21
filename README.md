<!--licence-->
<a href="./LICENSE">
    <img src="https://img.shields.io/badge/-MIT Licence-5674bc.svg?">
</a>
<br>
<br>
<!--logo-->
<p align="center">
<img width="70"  src = "./web_resource/kabosu_icon.png" />
</p>

[日本語](./web_resource/README_JA.md)

## kabosu-core ~Japanese preprocesser for TTS~


### Terms for uses (without obligation)
Please don't use for piracy, deepfake, other crimes.  
Use only authorized voices or your own voice

### To all companies
If you would like to use this software, please contact us.
We suport you.

kabosu-coreがもともと
kanalyzier + yomikata + jpreprocess + pyopenjtalk-plus + original nomarizer

kabosu-plusはkabosu-coreのコードを流用した、kabosu-coreのラッパーに
gplから引き継げるライセンスのライブラリを入れたものです。
style-bert-vits2のコードなどが含まれています。
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