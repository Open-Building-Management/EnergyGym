"""banque de modèles R1C1 et de paramètres généraux

La configuration nord est une sorte de hangar
dans lequel on chauffe de petits bureaux
avec un système de chauffage qui permet de remonter vite en température

R en K/W représente l'isolation du bâtiment

C en J/K représente l'inertie du bâtiment
"""
import random
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

LIST1 = ["tertiaire_peu_isolé", "tertiaire", "cells"]
LIST2 = [*LIST1, "nord_10_01_2022"]
FAST = ["unreal", "extremeb", "extreme", "nord_10_01_2022"]
SLOW = ["nord_10_01_2022_noreduce", *LIST1, "bloch", "bloch1"]
ALL = list(MODELS.keys())

NAMES = [*MODELS.keys(), "all", "list1", "list2", "fast", "slow", "synth"]

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

def generate(verbose=False, bank_name="slow", **kwargs):
    """RC generator"""
    rc_min = kwargs.get("rc_min", 50)
    rc_max = kwargs.get("rc_max", 100)
    try:
        bank = globals()[bank_name.upper()]
        model_name = random.choice(bank)
        _r_ = MODELS[model_name]["R"]
        _c_ = MODELS[model_name]["C"]
    except Exception:
        while True:
            _r_ = random.randint(1, 9) * random.choice([1e-3, 1e-4])
            _c_ = random.randint(1, 9) * random.choice([1e+7, 1e+8, 1e+9])
            if rc_min == -1 and rc_max == -1:
                break
            if rc_min == -1 and _r_ * _c_ / 3600 <= rc_max:
                break
            if rc_max == -1 and rc_min <= _r_ * _c_ / 3600:
                break
            if rc_min <= _r_ * _c_ / 3600 <= rc_max:
                break
    if verbose:
        print(f'{_r_:.2e} K/W, {_c_:.2e} J/K')
        tcte = round(_r_ * _c_ / 3600)
        print(f'{tcte}')
    return {"R" : _r_, "C" : _c_}

def output_model(model):
    """model pretty print :-)
    with scientific notation for R and C
    """
    output = ""
    for key, val in model.items():
        if isinstance(val, float):
            unit = ""
            if key == "R":
                unit = " K/W"
            if key == "C":
                unit = " J/K"
            output = f'{output} \'{key}\' : {val:.2e}{unit},'
        else:
            output = f'{output} \'{key}\' : {val},'
    print(f'{{{output[1:-1]}}}')
