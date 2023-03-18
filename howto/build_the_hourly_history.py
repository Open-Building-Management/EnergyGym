""" construction de l'histoire passée horaire """
import numpy as np

# pylint: disable=C0103

INTERVAL = 900
CW = 1162.5 #Wh/m3/K
MAX_POWER = 5 * CW * 15


NBH = 5
HTI = 3600 // INTERVAL
PASTSIZE = int(1 + NBH * 3600 // INTERVAL)

# températures extérieures et puissances de chauffage
# sur la base du pas de discrétisation choisi par l'utilisateur
text_past = np.random.random(PASTSIZE)
print(text_past)
q_c_past = np.random.randint(0,2,size=PASTSIZE-1)*MAX_POWER
print(q_c_past)

# calcul des puissances de chauffage moyennes au pas horaire
result = np.zeros(NBH)
for i in range(NBH):
    pos = i * HTI
    val = np.mean(q_c_past[pos:pos+HTI])
    result[i] = val
print(result)
print(np.array([*result/MAX_POWER]))

# pour la température,
# on ne passe par par une valeur moyenne
# mais par un simple échantillonnage sur base horaire
indexes = np.arange(0, 1 + NBH * HTI, HTI)
print(indexes)
print(text_past[indexes])
