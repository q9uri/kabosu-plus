
from kabosu_plus.sbv2.nlp.multiringual.g2p import g2p
from kabosu_plus.sbv2.constants import Languages    

print(g2p(text = "포상은 열심히 한 아이에게만 주어지기 때문에 포상인 것입니다.", language_list=[Languages.KO]))