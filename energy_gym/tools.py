"""ouverture de réseaux neurones ou de timeseries de données"""
import os
import sys
import random
# pour l'autocompletion en ligne de commande
import readline
import glob
import matplotlib.pylab as plt
import tensorflow as tf
from PyFina import PyFina, getMeta
from .planning import tsToHuman, biosAgenda


def set_extra_params(model, **kwargs):
    """définit d'éventuels paramètres additionnels dans le modèle"""
    fields = ["action_space", "mean_prev", "autosize_max_power",
              "k", "k_step", "p_c", "vote_interval",
              "nbh", "nbh_forecast"]
    for field in fields:
        if field in kwargs and kwargs[field]:
            model[field] = kwargs[field]
    return model


def load(agent_path):
    """load tensorflow network"""
    # custom_objects est nécessaire pour charger certains réseaux
    # cf ceux entrainés sur le cloud, via les github actions
    try:
        agent = tf.keras.models.load_model(
            agent_path,
            compile=False,
            custom_objects={'Functional':tf.keras.models.Model}
        )
    except Exception:
        print("could not load - using tf.saved_model.load() API as a workaround !")
        agent = tf.saved_model.load(agent_path)
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
        if tirage not in frozen:
            frozen.append(tirage)
    return frozen


def simple_path_completer(text, state):
    """
    tab completer pour les noms de fichiers, chemins....
    """
    return list(glob.glob(text + '*'))[state]


def pick_name(name=None, autocomplete=True, question="nom du réseau ?"):
    """
    vérifie un chemin ou un nom de fichier fourni en argument ou saisi en autocomplétion par l'utilisateur
    """
    if name is None and autocomplete:
        readline.set_completer_delims('\t')
        readline.parse_and_bind("tab: complete")
        readline.set_completer(simple_path_completer)
        name = input(question)
        if not name:
            name = "RL.h5"

    saved_model = False
    if os.path.isdir(name):
        if os.path.isfile(f'{name}/saved_model.pb'):
            saved_model = True
    else:
        if ".h5" not in name:
            name = f'{name}.h5'
        if os.path.isfile(name):
            saved_model = True

    return name, saved_model


def get_feed(feedid, interval, path="/var/opt/emoncms/phpfina"):
    """
    étant donné un numéro de flux et un pas de discrétisation exprimé en secondes

    récupère l'objet PyFina correspondant
    """
    meta = getMeta(feedid, path)
    full_length = meta["npoints"] * meta["interval"]
    tss = meta["start_time"]
    npoints = full_length // interval
    return PyFina(feedid, path, tss, interval, npoints)


def get_truth(circuit, visual_check):
    """
    DEPRECATED - only used by play which is only for old networks

    circuit : dictionnaire des paramètres du circuit

    récupère la vérité terrain :
    - température extérieure
    - agenda d'occupation
    - timestamps de début (tss) et de fin (tse)
    """

    feedid = circuit["Text"]
    path = circuit["dir"]
    schedule = circuit["schedule"]
    interval = circuit["interval"]
    wsize = circuit["wsize"]

    meta = getMeta(feedid, path)
    # durée du flux en secondes
    full_length = meta["npoints"] * meta["interval"]
    tss = meta["start_time"]
    tse = meta["start_time"] + full_length
    npoints = full_length // interval

    if tse - tss <= wsize*interval + 4*24*3600 :
        print("Vous n'aurez pas assez de données pour travailler : impossible de poursuivre")
        sys.exit()

    print("ENVIRONNEMENT")
    print(f'Démarrage : {meta["start_time"]}')
    print(f'Durée totale en secondes: {full_length}')
    print(f'Fin: {meta["start_time"]+full_length}')
    print(f'De {tsToHuman(tss)} à {tsToHuman(tse)}')
    print(f'Au pas de {interval} secondes, le nombre de points sera de {npoints}')
    print(f'Pour information, la durée d\'un épisode est de {wsize} intervalles')

    text = PyFina(feedid, path, tss, interval, npoints)

    agenda = biosAgenda(npoints, interval, tss, [], schedule=schedule)

    if visual_check:
        ax1 = plt.subplot(211)
        plt.title("vérité terrain")
        plt.plot(text, label="Text")
        plt.legend()
        plt.subplot(212, sharex=ax1)
        plt.plot(agenda, label="agenda")
        plt.legend()
        plt.show()

    return text, agenda
