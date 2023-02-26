"""banque de modèles R1C1 et de paramètres généraux
la configuration nord est une sorte de hangar
dans lequel on chauffe de petits bureaux
avec un système de chauffage qui permet de remonter vite en température
"""
import numpy as np

MODELS = {
    "unreal":  {"R" : 2e-04, "C": 2e+07},
    "extremeb":{"R" : 2e-04, "C": 9e+07},
    "extreme": {"R" : 2e-04, "C": 2e+08},
    "nord_10_01_2022": {"R" : 5.94419964e-04, "C" : 5.40132642e+07},
    "nord_10_01_2022_noreduce": {"R" : 1.50306273e-03, "C" : 4.42983318e+08},
    "tertiaire_peu_isolé":    {"R" : 5e-04, "C": 5e+08},
    "tertiaire":    {"R" : 1e-03, "C" : 9e+08},
    "bloch":   {"R" : 2.54061406e-04, "C" : 9.01650468e+08},
    "bloch1":  {"R" : 3.08814171e-04, "C" : 8.63446560e+08},
    "cells" :  {"R" : 2.59460660e-04, "C" : 1.31446233e+09}
}

TRAINING_LIST = ["tertiaire_peu_isolé", "tertiaire", "cells"]

PATH = "datas"
SCHEDULE = np.array([[7, 17], [7, 17], [7, 17], [7, 17], [7, 17], [-1, -1], [-1, -1]])
CW = 1162.5 #Wh/m3/K
# debit de 5m3/h et deltaT entre départ et retour de 15°C
MAX_POWER = 5 * CW * 15
TEXT_FEED = 1
REDUCE = 2

# série d'épisodes dont on veut avoir des replays
# les 3 premiers : froid
# les 3 suivants : très froids
# le dernier : mi-saison
COLD = [1577259140, 1605781940, 1608057140, 1610019140, 1612513940, 1611984740, 1633350740]
