
from kabosu_plus.sbv2.nlp.multiringual.g2p import g2p
from kabosu_plus.sbv2.constants import Languages    

print(g2p(text = "이 햇살처럼", language_list=[Languages.KO]))