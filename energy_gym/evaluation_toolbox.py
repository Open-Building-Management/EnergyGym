"""Evaluation toolbox"""
import random
import signal
import time
import copy
import math
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from .planning import tsToHuman, get_random_start, get_level_duration
from .heatgym import confort, presence, MODELRC

# nombre d'épisodes que l'on souhaite jouer
MAX_EPISODES = 900
PRIMO_AGENT_LAYERS = ['states', 'dense', 'dense_1']

def covering(tmin, tmax, tc, hh, ts, wsize, interval, occupation, xr=None):
    """
    permet l'habillage du graphique d'un épisode
    avec la zone de confort et les périodes d'occupation
    utilise la fonction Rectangle de matplotlib

    - tmin : température minimale
    - tmax : température maximale
    - tc : température de consigne
    - hh : demi-intervalle de la zone de confort
    - wsize : nombre de points dans l'épisode
    - interval : pas de temps
    - occupation : agenda d'occupation avec 4 jours supplémentaires
    pour calculer les nombres de pas jusqu'au prochain cgt d'occupation

    retourne les objets matérialisant les zones de confort et d'occupation
    """
    if xr is None:
        xrs = np.arange(ts, ts + wsize * interval, interval)
        xr = np.array(xrs, dtype='datetime64[s]')

    zone_confort = confort(xr, tc, hh)
    zones_occ = presence(xr, occupation, wsize, tmin, tmax, tc, hh)
    return xr, zone_confort, zones_occ


def sim(env, pos, tint0, nbh, action=1):
    """simulation suivant la méthode des trapèzes

    on est à la position pos dans env.text
    on veut calculer la température intérieure dans nbh heures
    soit en chauffant en continu soit sans chauffer

    si on veut prévoir le point suivant seulement, donc à pos+1,
    on doit donner à nbh la valeur env.text.step/3600

    la fonction retourne le tableau tint des températures simulées
    la valeur recherchée est tint[-1]
    """
    # nombre d'intervalles pour le calcul
    target = int(nbh * 3600 / env.text.step)
    # ON VEUT CONNAITRE LA TEMPERATURE INTERIEURE A pos+target
    text = env.text[pos: pos+target+1]
    tint = np.zeros(text.shape[0])
    tint[0] = tint0
    power = action * env.max_power
    for j in range(1, text.shape[0]):
        delta = env.cte * (power / env.model["C"] + text[j-1] / env.tcte)
        delta += power / env.model["C"] + text[j] / env.tcte
        tint[j] = tint[j-1] * env.cte + env.text.step * 0.5 * delta
    return tint


def play_hystnvacancy(env, pos, size, tint0, tc, hh, agenda=None):
    """joue la politique optimale sur un scénario d'intermittence
    avec un modèle déterministe contenu dans env

    Utilise soit l'agenda fourni en paramètre, soit celui de l'environnement

    Retourne un tableau de 2 colonnes et de size lignes
    colonne 1 : intensité de chauffage
    colonne 2 : température intérieure
    """
    # how many hour(s) is an interval ?
    # if text.step is 1800, ith will be 0.5
    ith = env.text.step / 3600
    datas = np.zeros((size, 2))
    datas[0, 1] = tint0
    if agenda is None:
        agenda = env.agenda[pos: pos+size+4*24*3600//env.text.step]
    # doit-on mettre en route le chauffage à l'étape 0 ?
    if agenda[0] == 0:
        nbh = get_level_duration(agenda, 0) * ith
        tint_sim = sim(env, pos, tint0, nbh)
        action = tint_sim[-1] <= tc
    else:
        action = tint0 <= tc
    datas[0, 0] = action
    # itération
    for i in range(1, size):
        #  calcul de la température à l'étape i
        tint_sim = sim(env, pos+i-1, datas[i-1, 1], ith, action)
        datas[i, 1] = tint_sim[-1]
        if agenda[i] == 0:
            # hors occupation : simulation à la cible
            # vu qu'on chauffe tt le temps, on ne précise pas action !
            nbh = get_level_duration(agenda, i) * ith
            tint_sim = sim(env, pos+i, datas[i, 1], nbh)
            action = tint_sim[-1] <= tc
        else:
            # hystérésis classique
            if datas[i, 1] > tc + hh or datas[i, 1] < tc - hh :
                action = datas[i, 1] <= tc
            else:
                # on est dans la fenêtre > on ne change rien :-)
                action = datas[i-1, 0]
        datas[i, 0] = action
    return datas

def stats(tc, tint, occ, interval):
    """basic stats on temperature datas"""
    idx = occ != 0
    tint_occ = tint[idx]
    # NumPy's built-in Fancy indexing
    inc = tint_occ[tint_occ < tc - 1]
    luxe = tint_occ[tint_occ > tc + 1]
    tocc_moy = round(np.mean(tint_occ), 2)
    nbinc = inc.shape[0] * interval / 3600
    nbluxe = luxe.shape[0] * interval / 3600
    return tocc_moy, nbinc, nbluxe

def get_config(agent):
    """
    extrait la configuration du réseau

    - lnames : liste des noms des couches
    - insize : taille de l'input
    - outsize : taille de la sortie
    """
    lnames = []
    for layer in agent.layers:
        lnames.append(layer.name)
    print(lnames)
    if lnames == PRIMO_AGENT_LAYERS:
        print("agent issu des expérimentations primitives")
    outlayer = agent.get_layer(name="output") if "output" in lnames else agent.get_layer(name=lnames[-1])
    inlayer = agent.get_layer(name="states") if "states" in lnames else agent.get_layer(name=lnames[0])
    outsize = outlayer.get_config()['units']
    try:
        insize = inlayer.get_config()["batch_input_shape"][1]
    except Exception :
        print("no input layer")
        insize = 4
    print(f'network input size {insize} output size {outsize}')
    return lnames, insize, outsize


class Environnement:
    """
    stocke les données décrivant l'environnement
    et offre des méthodes pour le caractériser

    - text : objet PyFina, vecteur numpy de température extérieure
             échantillonné selon le pas de discrétisation (interval)
    - agenda : vecteur numpy de l'agenda d'occupation échantillonné selon
             le même pas de discrétisation que text et de même taille que text
    - wsize : nombre d'intervalles constituant un épisode, l'épisode étant
              la métrique de base utilisé pour les entrainements et les replays.
    - tc : température de consigne / confort temperature set point (°C)
    - hh : demi-intervalle (en °C) pour le contrôle hysteresys
    - model : paramètres du modèle d'environnement - exemple : R=2e-4, C=2e8
    """
    def __init__(self, text, agenda, wsize, max_power, tc, hh, **model):
        self.text = text
        self.agenda = agenda
        self._tss = text.start
        self._tse = text.start + text.step * text.shape[0]
        self.interval = text.step
        self.wsize = wsize
        self.max_power = max_power
        self.tc = tc
        self.hh = hh
        print(f'environnement initialisé avec Tc={self.tc}, hh={self.hh}')
        self.model = model if model else MODELRC
        self.tcte = self.model["R"] * self.model["C"]
        self.cte = math.exp(-self.interval / self.tcte)
        self.pos = None
        self.tsvrai = None

    def set_start(self, ts=None):
        """
        1) tire un timestamp aléatoirement avant fin mai OU après début octobre

        2) fixe le timestamp à une valeur donnée, si ts est fourni,
           pour rejouer un épisode (ex : 1588701000)

        ts : unix timestamp (non requis)

        retourne la position dans la timeserie text et le timestamp correspondant
        """
        if ts is None:
            start = self._tss
            tse = self._tse
            end = tse - self.wsize * self.interval - 4*24*3600
            #print(tsToHuman(start),tsToHuman(end))
            # on tire un timestamp avant fin mai OU après début octobre
            ts = get_random_start(start, end, 10, 5)
        self.pos = (ts - self._tss) // self.interval
        self.tsvrai = self._tss + self.pos * self.interval
        print("*************************************")
        print(f'{ts} - {tsToHuman(ts)}')
        print(f'vrai={self.tsvrai} - {tsToHuman(self.tsvrai)}')

    def build_env(self, tint=None):
        """
        retourne le tenseur des données de l'épisode

        tint : valeur initiale de température intérieure
        Fournir un entier pour tint permet de fixer la température intérieure du premier point de l'épisode
        Si tint vaut None, un tirage aléatoire entre 17 et 20 est réalisé

        caractéristiques du tenseur de sortie

        - axe 0 = le temps
        - axe 1 = les paramètres pour décrire l'environnement

        3 paramètres physiques : qc, text et tint

        2 paramètres organisationnels :

        - temperature de consigne * occupation - si > 0 : bâtiment occupé,
        - nombre d'heures d'ici le changement d 'occupation
        """
        datas = np.zeros((self.wsize, 5))
        # condition initiale en température
        if isinstance(tint, (int, float)):
            datas[0, 2] = tint
        else:
            datas[0, 2] = random.randint(17, 20)
        # on connait Text (vérité terrain) sur toute la longueur de l'épisode
        datas[:, 1] = self.text[self.pos:self.pos+self.wsize]
        occupation = self.agenda[self.pos:self.pos+self.wsize+4*24*3600//self.interval]
        for i in range(self.wsize):
            datas[i, 4] = get_level_duration(occupation, i) * self.interval / 3600
        # consigne
        datas[:, 3] = self.tc * occupation[0:self.wsize]
        print(f'condition initiale : Text {datas[0, 1]:.2f} Tint {datas[0, 2]:.2f}')
        return datas

    def sim(self, datas, i):
        """calcule la température à l'étape i"""
        result = sim(self, self.pos+i-1, datas[i-1, 2],
                     self.text.step/3600, action=datas[i-1, 0]/self.max_power)
        return result[-1]


    def sim2target(self, datas, i):
        """
        on est à l'étape i et on veut calculer la température à l'ouverture des locaux,
        en chauffant dès à présent en permanence
        """
        return sim(self, self.pos+i, datas[i, 2], datas[i, 4])

    def play(self, datas):
        """
        à définir dans la classe fille pour jouer une stratégie de chauffe

        retourne le tenseur de données sources complété par le scénario de chauffage et la température intérieure simulée
        """


class Evaluate:
    """
    name : nom qui sera utilisé par la fonction close pour sauver le réseau et les graphiques

    name est compris par close comme un chemin
    et les graphes sont tjrs enregistrés dans le répertoire contenant l'agent

    _rewards sert à enregistrer le détail de la structure de la récompense sur un épisode
    exemple : partie confort, partie vote, partie energy
    à mettre à jour dans la classe fille dans la méthode reward()

    on parle de luxe si la température intérieure est supérieure à tc+hh
    on parle d'inconfort si la température intérieure est inférieure à tc-hh
    les colonnes de la matrice _stats sont les suivantes :
    - 0 : timestamp de l'épisode,
    - 1 à 4 : agent température intérieure moyenne, nb pts luxe, nb pts inconfort, consommation
    - 5 à 8 : modèle idem
    - 9 à 10 : récompense agent puis modèle : INUTILE
    """
    def __init__(self, name, env, agent, **params):
        self._n = params.get("N", MAX_EPISODES)
        self._k = params.get("k", 1)
        print(f'on va jouer {self._n} épisodes')
        self._name = name
        self._env = env
        self._modlabel = f'R={self._env.model["R"]:.2e} C={self._env.model["C"]:.2e}'
        self._agent = agent
        self._occupancy_agent = None
        print(f'métrique de l\'agent online {agent.metrics_names}')
        self._lnames, self._insize, self._outsize = get_config(agent)
        self._exit = False
        # numéro de l'épisode
        self._steps = 0
        self._policy = "agent"
        ini = defaultdict(lambda:np.zeros(self._env.wsize))
        self._rewards = {"agent":ini, "model":copy.deepcopy(ini)}
        self._stats = np.zeros((self._n, 11))
        self._multi_agent = False

    def set_occupancy_agent(self, agent):
        """add an occupancy agent such as an hystérésis"""
        self._multi_agent = True
        self._occupancy_agent = agent

    def _sig_handler(self, signum, frame):  # pylint: disable=unused-argument
        """gracefull shutdown"""
        print(f'signal de fermeture reçu {signum}')
        self._exit = True

    def stats(self, datas):
        """basic stats"""
        idx = datas[:, 3] != 0
        datas_occ = datas[idx, 2]
        inc = datas_occ[datas_occ[:] < self._env.tc - self._env.hh]
        luxe = datas_occ[datas_occ[:] > self._env.tc + self._env.hh]
        tocc_moy = round(np.mean(datas_occ[:]), 2)
        nbinc = inc.shape[0] * self._env.interval // 3600
        nbluxe = luxe.shape[0] * self._env.interval // 3600
        return tocc_moy, nbinc, nbluxe

    def play_gym(self, silent, ts=None, snapshot=False, tint=None, wsize=None):
        """joue un épisode de type semaine
        avec l'environnement gym"""
        tc = self._env.tc
        state = self._env.reset(ts=ts, tc_step=tc, tint=tint, wsize=wsize)
        while True:
            pos1 = self._env.i
            pos2 = self._env.pos + pos1
            if self._env.agenda[pos2] != 0 and self._multi_agent:
                hyststate = np.array([
                    self._env.text[pos2],
                    self._env.tint[pos1],
                    tc])
                result = self._occupancy_agent(hyststate.reshape(1, hyststate.shape[0]))
            else:
                result = self._agent(state.reshape(1, state.shape[0]))
            action = np.argmax(result)
            state, _, done, _ = self._env.step(action, tc_step=tc)
            if done:
                break
        tint = self._env.tint[:-1]
        occ = self._env.agenda[self._env.pos: self._env.pos+self._env.wsize]
        interval = self._env.text.step
        atocc_moy, anbinc, anbluxe = stats(tc, tint, occ, interval)
        optimal_solution = play_hystnvacancy(self._env,
                                             self._env.pos,
                                             self._env.wsize,
                                             tint[0], tc, 1)
        mtocc_moy, mnbinc, mnbluxe = stats(tc, optimal_solution[:,1], occ, interval)
        aconso = self._env.wsize - self._env.tot_eko
        mconso = np.sum(optimal_solution[:,0])
        aeko = 100 * self._env.tot_eko / self._env.wsize
        meko = 100 * (1 - np.mean(optimal_solution[:,0]))
        line = np.array([self._env.tsvrai,
                         atocc_moy, anbluxe, anbinc, aconso,
                         mtocc_moy, mnbluxe, mnbinc, mconso,
                         0, 0])
        self._stats[self._steps, :] = line
        if not silent or snapshot:
            label = f'EKO - modèle : {meko:.2f}% - agent : {aeko:.2f}%'
            label = f'{label} {self._modlabel}'
            label = f'{label}\n Tocc moyenne modèle : {mtocc_moy} agent : {atocc_moy}'
            label = f'{label}\n nb heures inconfort modèle : {mnbinc} agent : {anbinc}'
            self._env.render(stepbystep=False,
                             label=label,
                             extra_datas=optimal_solution,
                             snapshot=snapshot)


    def play(self, silent, ts=None, snapshot=False, tint=None):
        """
        joue un épisode

        retourne le tenseur des données de l'agent, au cas où on souhaite y vérifier un détail

        ts : int - timestamp que l'on veut rejouer
        si None, un tirage alétaoire est réalisé

        snapshot : boolean
        si snapshot est True, l'image de l'épisode n'est pas affichée
        et un fichier tiers utilisant la classe peut l'enregistrer

        tint : condition initiale de température intérieure
        """
        self._env.set_start(ts)
        adatas = self._env.build_env(tint=tint)
        wsize = adatas.shape[0]
        # on fait jouer le modèle et on calcule sa consommation énergétique
        mdatas = self._env.play(copy.deepcopy(adatas))
        mconso = int(np.sum(mdatas[1:, 0]) / 1000) * self._env.interval // 3600
        self._rewards["agent"].clear()
        self._rewards["model"].clear()
        cumularewards = []
        cumulmrewards = []
        areward = 0
        mreward = 0

        for i in range(1, wsize):
            if self._lnames == PRIMO_AGENT_LAYERS:
                if self._insize == 5 :
                    state = np.zeros((self._insize))
                    state[0] = adatas[i-1, 2]
                    state[1] = adatas[i-1, 1]
                    state[2] = 1 if adatas[i-1, 3] > 0 else 0
                    state[3] = adatas[i-1, 4]
                    state[4] = adatas[i-1, 3]
                else:
                    # on permute Tint et Text car les agents jusque début 2021 prenaient Tint en premier....
                    # on pourrait utiliser np.array([ adatas[i-1,2], adatas[i-1,1], adatas[i-1,3], adatas[i-1,4] ])
                    # mais le slicing donne un code plus lisible et plus court :-)
                    reorder = [2, 1, 3, 4]
                    state = adatas[i-1, reorder[0:self._insize]]
            else:
                state = adatas[i-1, 1:self._insize + 1]
            agent = self._agent
            if self._multi_agent:
                agent = self._occupancy_agent if state[2] != 0 else self._agent
            prediction_brute = agent(state.reshape(1, self._insize))
            action = np.argmax(prediction_brute)
            adatas[i-1, 0] = action * self._env.max_power
            # on peut désormais calculer la récompense à l'étape i-1
            self._policy = "agent"
            areward += self.reward(adatas, i-1)
            self._policy = "model"
            mreward += self.reward(mdatas, i-1)
            cumularewards.append(areward)
            cumulmrewards.append(mreward)
            # calcul de la température à l'état suivant
            adatas[i, 2] = self._env.sim(adatas, i)
        aconso = int(np.sum(adatas[1:, 0]) / 1000) * self._env.interval // 3600
        print(f'récompense agent {areward:.2f} récompense modèle {mreward:.2f}')

        # on ne prend pas le premier point de température intérieure car c'est une condition initiale arbitraire
        atocc_moy, anbinc, anbluxe = self.stats(adatas[1:, :])
        mtocc_moy, mnbinc, mnbluxe = self.stats(mdatas[1:, :])
        line = np.array([self._env.tsvrai,
                         atocc_moy, anbluxe, anbinc, aconso,
                         mtocc_moy, mnbluxe, mnbinc, mconso,
                         areward, mreward])
        #print(line)
        self._stats[self._steps, :] = line

        if not silent or snapshot:
            covargs = []
            covargs.append(min(np.min(mdatas[:, 2]), np.min(adatas[:, 2])))
            covargs.append(max(np.max(mdatas[:, 2]), np.min(adatas[:, 2])))
            covargs.append(self._env.tc)
            covargs.append(self._env.hh)
            covargs.append(self._env.tsvrai)
            covargs.append(wsize)
            covargs.append(self._env.interval)
            covargs.append(self._env.agenda[self._env.pos:self._env.pos + wsize + 4*24*3600 // self._env.interval])
            xr, zone_confort, zones_occ = covering(*covargs)  # pylint: disable=no-value-for-parameter

            title = f'épisode {self._steps}'
            title = f'{title} - {self._env.tsvrai} {tsToHuman(self._env.tsvrai)}'
            title = f'{title} {self._modlabel}'
            title = f'{title}\n conso Modèle {mconso} Agent {aconso}'
            title = f'{title}\n Tocc moyenne modèle : {mtocc_moy} agent : {atocc_moy}'
            title = f'{title}\n nb heures inconfort modèle : {mnbinc} agent : {anbinc}'

            if snapshot:
                plt.figure(figsize=(20, 10))

            nbg = 411
            ax1 = plt.subplot(nbg)
            plt.title(title, fontsize=8)
            ax1.add_patch(zone_confort)
            for occ in zones_occ:
                ax1.add_patch(occ)
            plt.ylabel("Temp. intérieure °C")
            plt.plot(xr, mdatas[:, 2], color="orange", label="TintMod")
            plt.plot(xr, adatas[:, 2], color="black", label="TintAgent")
            plt.legend(loc='upper left')

            ax1.twinx()
            plt.ylabel("Temp. extérieure °C")
            plt.plot(xr, mdatas[:, 1], color="blue", label="Text")
            plt.legend(loc='upper right')

            nbg += 1
            ax3 = plt.subplot(nbg, sharex=ax1)
            plt.ylabel("Conso W agent")
            plt.plot(xr, adatas[:, 0], color="black", label="consoAgent")
            plt.legend(loc='upper left')

            ax4 = ax3.twinx()
            plt.ylabel("cum.reward agent")
            ypos = np.zeros(wsize)
            for rewtyp in self._rewards["agent"]:
                ax4.fill_between(xr, ypos, ypos + self._rewards["agent"][rewtyp],
                                 alpha=0.6, label=f'agent {rewtyp}')
                ypos = ypos + self._rewards["agent"][rewtyp]
            #plt.plot(xr[1:], cumularewards, color="black", label="agent")
            plt.legend(loc='upper right')

            nbg += 1
            ax5 = plt.subplot(nbg, sharex=ax1)
            plt.ylabel("Conso W modèle")
            plt.plot(xr, mdatas[:, 0], color="orange", label="consoMod")
            plt.legend(loc='upper left')

            ax6 = ax5.twinx()
            plt.ylabel("cum.reward mod")
            ypos = np.zeros(wsize)
            for rewtyp in self._rewards["model"]:
                ax6.fill_between(xr, ypos, ypos + self._rewards["model"][rewtyp],
                                 alpha=0.6, label=f'model {rewtyp}')
                ypos = ypos + self._rewards["model"][rewtyp]
            #plt.plot(xr[1:], cumulmrewards, color="orange", label="mod")
            plt.legend(loc='upper right')

            # à enlever si on veut alléger le graphique
            nbg += 1
            ax7 = plt.subplot(nbg, sharex=ax1)
            plt.ylabel("°C")
            plt.plot(xr, mdatas[:, 3], label="consigne")
            plt.legend(loc='upper left')
            ax7.twinx()
            plt.ylabel("nb steps > cgt occ.")
            plt.plot(xr, mdatas[:, 4], 'o', markersize=1, color="red")

            if not snapshot:
                plt.show()
        return adatas

    def reward(self, datas, i):
        """fonction récompense à définir dans la classe fille"""

    def run(self, silent=False, tint=None):
        """boucle d'exécution"""
        signal.signal(signal.SIGINT, self._sig_handler)
        signal.signal(signal.SIGTERM, self._sig_handler)

        while not self._exit:
            if self._steps >= self._n - 1:
                self._exit = True

            self.play(silent=silent, tint=tint)
            self._steps += 1

            time.sleep(0.1)

    def run_gym(self, silent=False, wsize=None):
        """boucle d'exécution"""
        signal.signal(signal.SIGINT, self._sig_handler)
        signal.signal(signal.SIGTERM, self._sig_handler)

        while not self._exit:
            if self._steps >= self._n - 1:
                self._exit = True

            self.play_gym(silent=silent, wsize=wsize)
            self._steps += 1

            time.sleep(0.1)

    def close(self, suffix=None):
        """
        enregistre les statistiques (csv + png) si on est arrivé au bout du nombre d'épisodes

        suffix, s'il est fourni, sert dans la construction de(s) nom(s) de fichier(s),
        pour préciser par ex. le type de politique optimale jouée par le modèle
        """

        stats_moy = np.mean(self._stats, axis=0).round(1)

        print("leaving the game")
        # enregistrement des statistiques du jeu
        # uniquement si on est allé au bout des épisodes - pas la peine de sauver des figures vides
        # on utilise le suffixe pour indiquer le mode de jeu du modèle
        if self._steps == self._n :

            title = f'modèle {self._modlabel}'
            #' jouant la politique optimale {suffix}\n' if suffix is not None else ""
            title = f'{title} Conso moyenne agent : {stats_moy[4]} / Conso moyenne modèle : {stats_moy[8]}\n'

            pct = round(100*(stats_moy[8]-stats_moy[4])/stats_moy[8], 2)
            title = f'{title} Pourcentage de gain agent : {pct} %'

            plt.figure(figsize=(20, 10))
            ax1 = plt.subplot(411)
            plt.title(title)
            label = "température moyenne occupation"
            plt.plot(self._stats[:, 1], color="blue", label=f'{label} agent')
            plt.plot(self._stats[:, 5], color="red", label=f'{label} modèle')
            plt.legend()

            plt.subplot(412, sharex=ax1)
            label = f'nombre heures > {self._env.tc + 1}°C'
            plt.plot(self._stats[:, 2], color="blue", label=f'{label} agent')
            plt.plot(self._stats[:, 6], color="red", label=f'{label} modèle')
            plt.legend()

            plt.subplot(413, sharex=ax1)
            label = f'nombre heures < {self._env.tc - 1}°C'
            plt.plot(self._stats[:, 3], color="blue", label=f'{label} agent')
            plt.plot(self._stats[:, 7], color="red", label=f'{label} modèle')
            plt.legend()

            plt.subplot(414, sharex=ax1)
            label = "récompense cumulée"
            plt.plot(self._stats[:, 9], color="blue", label=f'{label} agent')
            plt.plot(self._stats[:, 10], color="red", label=f'{label} modèle')
            plt.legend()

            ts = time.time()
            now = tsToHuman(ts, fmt="%Y_%m_%d_%H_%M")
            label = f'played_{suffix}' if suffix is not None else "played"
            # si on est arrivé jusqu'içi,
            # c'est que l'utilisateur a chargé un réseau de neurones
            # donc pas la peine de faire des tests compliqués
            if ".h5" not in self._name:
                name = f'{self._name}/{label}_{now}'
            else:
                name = f'{self._name.replace(".h5","")}_{label}_{now}'
            plt.savefig(name)
            header = "ts"
            for player in ["agent", "modèle"]:
                header = f'{header},{player}_Tintmoy,{player}_nbpts_luxe'
                header = f'{header},{player}_nbpts_inconfort,{player}_conso'
            np.savetxt(f'{name}.csv', self._stats, delimiter=',', header=header)

        plt.close()

        return stats_moy
