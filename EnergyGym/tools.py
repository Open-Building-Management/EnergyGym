"""
outils et méthodes autour des classes de RLtoolbox

version beta : work in progress

circuit : dictionnaire des paramètres du circuit / zone de bâtiment

exemple :
```
import numpy as np
schedule = np.array([ [7,17], [7,17], [7,17], [7,17], [7,17], [-1,-1], [-1,-1] ])
circuit = {"Text":1, "dir": "/var/opt/emoncms/phpfina",
           "schedule": schedule, "interval": 3600, "wsize": 1 + 8*24}
```
"""
import os
import sys
# pour l'autocompletion en ligne de commande
import readline
import glob
import matplotlib.pylab as plt
from PyFina import PyFina, getMeta
from .planning import tsToHuman, biosAgenda


def simple_path_completer(text, state):
    """
    tab completer pour les noms de fichiers, chemins....
    """
    return list(glob.glob(text + '*'))[state]


def pick_name(name=None, autocomplete=True):
    """
    vérifie un chemin ou un nom de fichier fourni en argument ou saisi en autocomplétion par l'utilisateur
    """
    if name is None and autocomplete:
        readline.set_completer_delims('\t')
        readline.parse_and_bind("tab: complete")
        readline.set_completer(simple_path_completer)
        name = input("nom du réseau ?")
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
    npoints =  full_length // interval
    return PyFina(feedid, path, tss, interval, npoints)


def get_truth(circuit, visual_check):
    """
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
    npoints =  full_length // interval

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
