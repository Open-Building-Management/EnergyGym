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

import matplotlib.pylab as plt
from PyFina import PyFina, getMeta
from .planning import tsToHuman, biosAgenda
import os

# pour l'autocompletion en ligne de commande
import readline
import glob
def simplePathCompleter(text,state):
    """
    tab completer pour les noms de fichiers, chemins....
    """
    line   = readline.get_line_buffer().split()

    return [x for x in glob.glob(text+'*')][state]

def pickName(name = None, autocomplete = True):
    """
    vérifie un chemin ou un nom de fichier fourni en argument ou saisi en autocomplétion par l'utilisateur
    """
    if name is None and autocomplete:
        readline.set_completer_delims('\t')
        readline.parse_and_bind("tab: complete")
        readline.set_completer(simplePathCompleter)
        name = input("nom du réseau ?")
        if not name:
            name = "RL.h5"

    savedModel = False
    if os.path.isdir(name):
        if os.path.isfile("{}/saved_model.pb".format(name)):
            savedModel = True
    else:
        if ".h5" not in name:
            name = "{}.h5".format(name)
        if os.path.isfile(name):
            savedModel = True

    return name, savedModel

def getFeed(feedid, interval, dir="/var/opt/emoncms/phpfina"):
    """
    étant donné un numéro de flux et un pas de discrétisation exprimé en secondes

    récupère l'objet PyFina correspondant
    """
    meta = getMeta(feedid, dir)
    fullLength = meta["npoints"] * meta["interval"]
    _tss = meta["start_time"]
    _tse = meta["start_time"]+fullLength
    npoints =  fullLength // interval
    return PyFina(feedid, dir, _tss, interval, npoints)


def getTruth(circuit, visualCheck):
    """
    circuit : dictionnaire des paramètres du circuit

    récupère la vérité terrain :
    - température extérieure
    - agenda d'occupation
    - timestamps de début(_tss) et de fin(_tse)
    """

    feedid = circuit["Text"]
    dir = circuit["dir"]
    schedule = circuit["schedule"]
    interval = circuit["interval"]
    wsize = circuit["wsize"]

    meta = getMeta(feedid, dir)
    # durée du flux en secondes
    fullLength = meta["npoints"] * meta["interval"]
    _tss = meta["start_time"]
    _tse = meta["start_time"]+fullLength
    npoints =  fullLength // interval

    if _tse - _tss <= wsize*interval + 4*24*3600 :
        print("Vous n'aurez pas assez de données pour travailler : impossible de poursuivre")
        sys.exit()

    print("| ____|_ ____   _(_)_ __ ___  _ __  _ __ ___   ___ _ __ | |_")
    print("|  _| | '_ \ \ / / | '__/ _ \| '_ \| '_ ` _ \ / _ \ '_ \| __|")
    print("| |___| | | \ V /| | | | (_) | | | | | | | | |  __/ | | | |_")
    print("|_____|_| |_|\_/ |_|_|  \___/|_| |_|_| |_| |_|\___|_| |_|\__|")

    print("Démarrage : {}".format(meta["start_time"]))
    print("Durée totale en secondes: {}".format(fullLength))
    print("Fin: {}".format(meta["start_time"]+fullLength))
    print("De {} à {}".format(tsToHuman(_tss),tsToHuman(_tse)))
    print("au pas de {} secondes, le nombre de points sera de {}".format(interval,npoints))
    print("pour information, la durée d'un épisode est de {} intervalles".format(wsize))

    Text = PyFina(feedid, dir, _tss, interval, npoints)

    agenda = biosAgenda(npoints, interval, _tss, [], schedule=schedule)

    if visualCheck:
        ax1 = plt.subplot(211)
        plt.title("vérité terrain")
        plt.plot(Text, label="Text")
        plt.legend()
        plt.subplot(212, sharex=ax1)
        plt.plot(agenda, label="agenda")
        plt.legend()
        plt.show()

    return Text, agenda
