"""banque de modèles R1C1 et de paramètres généraux
la configuration nord est une sorte de hangar
dans lequel on chauffe de petits bureaux
avec un système de chauffage qui permet de remonter vite en température
"""
import random
import numpy as np
import tensorflow as tf

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

def set_extra_params(model, **kwargs):
    """définit d'éventuels paramètres additionnels dans le modèle"""
    fields = ["action_space", "k", "p_c", "vote_interval", "nbh", "nbh_forecast"]
    for field in fields:
        if field in kwargs and kwargs[field]:
            model[field] = kwargs[field]
    return model

def load(agent_path):
    """load tensorflow network"""
    # custom_objects est nécessaire pour charger certains réseaux
    # cf ceux entrainés sur le cloud, via les github actions
    agent = tf.keras.models.load_model(
        agent_path,
        compile=False,
        custom_objects={'Functional':tf.keras.models.Model}
    )
    return agent

def freeze(nb_off):
    """retourne le tableau des numéros des jours
    chomés dans la semaine, en plus du week-end

    nb_off : nombre de jours chomés à injecter
    """
    days = [0, 4] if nb_off == 1 else [0, 1, 2, 3, 4]
    frozen = []
    for _ in range(nb_off):
        tirage = random.choice(days)
        if tirage not in holidays:
            frozen.append(tirage)
    return frozen
