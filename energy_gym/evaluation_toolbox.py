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
from .heatgym import confort, presence, MODELRC, sim, play_hystnvacancy

# nombre d'épisodes que l'on souhaite jouer
MAX_EPISODES = 900
PRIMO_AGENT_LAYERS = ['states', 'dense', 'dense_1']



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
    DEPRECATED

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
    DEPRECATED : UTILISER L'ENVIRONNEMENT GYM

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

class EvaluateGym:
    """base evaluation class

    Joue des épisodes et nourrit une matrice _stats avec des données statistiques
    """
    def __init__(self, name, env, agent, **params):
        """
        name : chemin utilisé par close pour sauver le réseau et les graphiques
        en général, name est le chemin du dossier contenant l'agent

        env : ready to use gym environment

        agent : réseau de neurones chargé en mémoire

        on parle de luxe si la température intérieure est supérieure à tc+hh

        on parle d'inconfort si la température intérieure est inférieure à tc-hh

        structure de la matrice _stats :
        colonne 0 = timestamp de l'épisode,
        colonnes 1 à 4 = temp. int. moyenne, nbh luxe, nbh inconfort, conso pour l'agent,
        colonnes 5 à 8 = les mêmes grandeurs pour le modèle
        """
        self._n = params.get("N", MAX_EPISODES)
        print(f'on va jouer {self._n} épisodes')
        self._name = name
        self._env = env
        self._modlabel = self._gen_mod_label()
        self._agent = agent
        self._occupancy_agent = None
        self._exit = False
        # numéro de l'épisode
        self.nb_episode = 0
        self._stats = np.zeros((self._n, 11))
        self._multi_agent = False

    def _gen_mod_label(self):
        """return a string with the model electrical parameters"""
        return f'R={self._env.model["R"]:.2e} C={self._env.model["C"]:.2e}'

    def update_model(self, model):
        """model live update"""
        self._env.update_model(model)
        self._modlabel = self._gen_mod_label()

    def get_episode_params(self):
        """return tint[0] and tsvrai"""
        return self._env.tint[0], self._env.tsvrai

    def set_occupancy_agent(self, agent):
        """add an occupancy agent such as an hystérésis"""
        self._multi_agent = True
        self._occupancy_agent = agent

    def _sig_handler(self, signum, frame):  # pylint: disable=unused-argument
        """gracefull shutdown"""
        print(f'signal de fermeture reçu {signum}')
        self._exit = True

    def play_base(self, ts=None, tint=None, wsize=None, same_tc_ono=True):
        """fait jouer à l'agent un épisode de type semaine
        avec l'environnement gym,
        calcule la solution optimale,
        établit les statistiques et les enregistre dans _stats

        ts : int - timestamp que l'on veut rejouer.
        si None, un tirage aléatoire est réalisé

        tint : condition initiale de température intérieure
        si on veut la fixer

        wsize : nombre de points dans l'épisode

        retourne la solution optimale mais n'affiche pas le replay
        """
        tc = self._env.tc
        tc_step = tc if same_tc_ono else None
        state = self._env.reset(ts=ts, tc_step=tc_step, tint=tint, wsize=wsize)
        while True:
            pos1 = self._env.i
            pos2 = self._env.pos + pos1
            coeff = 1
            if self._env.agenda[pos2] != 0 and self._multi_agent:
                hyststate = np.array([
                    self._env.text[pos2],
                    self._env.tint[pos1],
                    tc])
                coeff = self._env.action_space.n - 1
                result = self._occupancy_agent(hyststate.reshape(1, *hyststate.shape))
            else:
                result = self._agent(state.reshape(1, *state.shape))
            action = np.argmax(result) * coeff
            state, _, done, _ = self._env.step(action, tc_step=tc_step)
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
        mtocc_moy, mnbinc, mnbluxe = stats(tc, optimal_solution[:, 1], occ, interval)
        aconso = self._env.wsize + 1 - self._env.tot_eko
        mconso = np.sum(optimal_solution[:, 0])
        line = np.array([self._env.tsvrai,
                         atocc_moy, anbluxe, anbinc, aconso * interval / 3600,
                         mtocc_moy, mnbluxe, mnbinc, mconso * interval / 3600,
                         self._env.model["R"], self._env.model["C"]])
        self._stats[self.nb_episode, :] = line
        return optimal_solution

    def play_gym(self, ts=None, snapshot=False, tint=None, wsize=None, same_tc_ono=True):
        """
        render the replay

        snapshot : si True, le replay est construit mais pas affiché.
        Un fichier tiers utilisant la classe peut donc l'enregistrer
        """
        optimal_solution = self.play_base(ts=ts, tint=tint, wsize=wsize,
                                          same_tc_ono=same_tc_ono)
        print("agent eko", self._env.tot_eko)
        aeko = 100 * self._env.tot_eko / (self._env.wsize + 1)
        meko = 100 * (1 - np.mean(optimal_solution[:, 0]))
        label = f'EKO - modèle : {meko:.2f}% - agent : {aeko:.2f}%'
        max_power = round(self._env.max_power * 1e-3)
        label = f'{label} {self._modlabel} max power {max_power} kW'
        label = f'{label}\n Tocc moyenne'
        label = f'{label} modèle : {self._stats[self.nb_episode, 5]}'
        label = f'{label} agent : {self._stats[self.nb_episode, 1]}'
        label = f'{label}\n nb heures inconfort'
        label = f'{label} modèle : {self._stats[self.nb_episode, 7]}'
        label = f'{label} agent : {self._stats[self.nb_episode, 3]}'
        self._env.render(stepbystep=False,
                         label=label,
                         extra_datas=optimal_solution,
                         snapshot=snapshot)

    def run_gym(self, silent=False, wsize=None):
        """boucle d'exécution

        silent : si True, ne construit pas les images des replays
        et produit des stats

        wsize : nombre de points dans l'épisode (facultatif)
        """
        signal.signal(signal.SIGINT, self._sig_handler)
        signal.signal(signal.SIGTERM, self._sig_handler)
        while not self._exit:
            if self.nb_episode >= self._n - 1:
                self._exit = True
            if silent:
                self.play_base(wsize=wsize)
            else:
                self.play_gym(wsize=wsize)
            self.nb_episode += 1
            time.sleep(0.1)

    def close(self, suffix=None, random_model=False):
        """
        enregistre les statistiques (csv + png) si on est arrivé au bout du nombre d'épisodes

        suffix, s'il est fourni, sert dans la construction des noms de fichiers
        """
        stats_moy = np.mean(self._stats, axis=0).round(1)
        print("leaving the game")
        # enregistrement des statistiques du jeu
        # uniquement si on est allé au bout des épisodes
        # pas la peine de sauver des figures vides
        if self.nb_episode == self._n :
            title = ""
            if not random_model:
                max_power = round(self._env.max_power * 1e-3)
                title = f'modèle {self._modlabel} max power {max_power} kW'
            title = f'{title} Conso moyenne agent : {stats_moy[4]}'
            title = f'{title} / Conso moyenne modèle : {stats_moy[8]}\n'

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
            label = f'nb heures > {self._env.tc + 1}°C'
            plt.plot(self._stats[:, 2], color="blue", label=f'{label} agent')
            plt.plot(self._stats[:, 6], color="red", label=f'{label} modèle')
            plt.legend()

            plt.subplot(413, sharex=ax1)
            label = f'nb heures < {self._env.tc - 1}°C'
            plt.plot(self._stats[:, 3], color="blue", label=f'{label} agent')
            plt.plot(self._stats[:, 7], color="red", label=f'{label} modèle')
            plt.legend()

            plt.subplot(414, sharex=ax1)
            label = "conso (nb heures à puissance max)"
            plt.plot(self._stats[:, 4], color="blue", label=f'{label} agent')
            plt.plot(self._stats[:, 8], color="red", label=f'{label} modèle')
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
            header = f'{header},R_K/W,C_J/K'
            np.savetxt(f'{name}.csv', self._stats, delimiter=',', header=header)
        plt.close()
        return stats_moy

class Evaluate(EvaluateGym):
    """evaluation class"""
    def __init__(self, name, env, agent, **params):
        """
        DEPRECATED - use EvaluateGym

        only used by play which is for old networks

        _rewards sert à enregistrer le détail de la structure de la récompense sur un épisode
        exemple : partie confort, partie vote, partie energy
        à mettre à jour dans la classe fille dans la méthode reward()

        _stats contient les récompenses agent et modèle en position 9 à 10 :
        INUTILE
        """
        super().__init__(name, env, agent, **params)
        self._k = params.get("k", 1)
        print(f'métrique de l\'agent online {agent.metrics_names}')
        self._lnames, self._insize, self._outsize = get_config(agent)
        self._policy = "agent"
        ini = defaultdict(lambda:np.zeros(self._env.wsize))
        self._rewards = {"agent":ini, "model":copy.deepcopy(ini)}
        self._stats = np.zeros((self._n, 11))

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

    def play(self, silent, ts=None, snapshot=False, tint=None):
        """
        joue un épisode

        retourne le tenseur des données de l'agent, au cas où on souhaite y vérifier un détail
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
            prediction_brute = agent(state.reshape(1, *state.shape))
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
        self._stats[self.nb_episode, :] = line

        if not silent or snapshot:
            tmin = min(np.min(mdatas[:, 2]), np.min(adatas[:, 2]))
            tmax = max(np.max(mdatas[:, 2]), np.min(adatas[:, 2]))
            tc = self._env.tc
            hh = self._env.hh
            ts = self._env.tsvrai
            interval = self._env.interval
            occupation = self._env.agenda[self._env.pos:self._env.pos + wsize + 4*24*3600 // self._env.interval]
            xrs = np.arange(ts, ts + wsize * interval, interval)
            xr = np.array(xrs, dtype='datetime64[s]')
            zone_confort = confort(xr, tc, hh)
            zones_occ = presence(xr, occupation, wsize, tmin, tmax, tc, hh)

            title = f'épisode {self.nb_episode}'
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
            if self.nb_episode >= self._n - 1:
                self._exit = True
            self.play(silent=silent, tint=tint)
            self.nb_episode += 1
            time.sleep(0.1)
