"""banque de modèles R1C1
la configuration nord est une sorte de hangar
dans lequel on chauffe de petits bureaux
avec un système de chauffage qui permet de remonter vite en température
"""
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

TRAINING_LIST = ["nord_10_01_2022", "tertiaire_peu_isolé", "tertiaire", "cells"]
