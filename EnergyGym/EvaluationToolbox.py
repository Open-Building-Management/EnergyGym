"""
Evaluation toolbox
"""
# nombre d'épisodes que l'on souhaite jouer
MAX_EPISODES = 900

# modèle par défault de type R1C1 obtenues par EDW avec les données de Marc Bloch
modelRC = {"R": 2.54061406e-04, "C": 9.01650468e+08}

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random
import signal
import time
import copy
import math

from .planning import tsToHuman, getRandomStart, getLevelDuration
from .HeatGym import covering

def R1C1(step, R, C, Qc, Text, Tint, cte=None):
    """
    calcule le point suivant avec la méthode des trapèzes
    Text et Qc sont des vecteurs de taille 2 contenant les valeurs à t et t+step
    """
    if cte is None:
        cte = exp(-step/(R*C))
    delta = cte * ( Qc[0] / C + Text[0] / (R*C) ) + Qc[0] / C + Text[1] / (R*C)
    x = Tint * cte + step * 0.5 * delta
    return x

def R1C1sim(step, R, C, Qc, Text, T0, cte=None):
    """
    simulation utilisant le simple calcul du point suivant
    """
    n = Text.shape[0]
    x = np.zeros(n)
    x[0] = T0
    for i in range(1,n):
        x[i] = R1C1(step, R, C, Qc[i-1:i+1], Text[i-1:i+1], x[i-1], cte=cte)
    return x

def getConfig(agent):
    """
    extrait la configuration du réseau

    - LNames : liste des noms des couches
    - inSize : taille de l'input
    - outSize : taille de la sortie
    """
    LNames = []
    for layer in agent.layers:
        LNames.append(layer.name)
    print(LNames)
    if LNames == ['states', 'dense', 'dense_1']:
        print("agent issu des expérimentations primitives")
    outlayer = agent.get_layer(name="output") if "output" in LNames else agent.get_layer(name=LNames[-1])
    inlayer = agent.get_layer(name="states") if "states" in LNames else agent.get_layer(name=LNames[0])
    outSize = outlayer.get_config()['units']
    try:
        inSize = inlayer.get_config()["batch_input_shape"][1]
    except Exception as e:
        print("no input layer")
        inSize = 4
    print("network input size {} output size {}".format(inSize, outSize))
    return LNames, inSize, outSize

class Environnement:
    """
    stocke les données décrivant l'environnement et offre des méthodes pour le caractériser

    - Text : objet PyFina, vecteur numpy de température extérieure échantillonné selon le pas de discrétisation (interval)
    - agenda : vecteur numpy de l'agenda d'occupation échantillonné selon le même pas de discrétisation que Text et de même taille que Text
    - wsize : nombre d'intervalles constituant un épisode, l'épisode étant la métrique de base utilisé pour les entrainements et les replays.
    - Tc : température de consigne / confort temperature set point (°C)
    - hh : demi-intervalle (en °C) pour le contrôle hysteresys
    - model : paramètres du modèle d'environnement - exemple : R=2e-4, C=2e8
    """
    def __init__(self, Text, agenda, wsize, max_power, Tc, hh, **model):
        self._Text = Text
        self._agenda = agenda
        self._tss = Text.start
        self._tse = Text.start + Text.step * Text.shape[0]
        self._interval = Text.step
        self._wsize = wsize
        self._max_power = max_power
        self._Tc = Tc
        self._hh = hh
        print("environnement initialisé avec Tc={}, hh={}".format(self._Tc, self._hh))
        self._model = modelRC
        if model:
            self._model = model
        self._cte = math.exp(-self._interval/(self._model["R"]*self._model["C"]))

    def setStart(self, ts=None):
        """

        1) tire un timestamp aléatoirement avant fin mai OU après début octobre

        2) fixe le timestamp à une valeur donnée, si ts est fourni, pour rejouer un épisode (ex : 1588701000)

        ts : unix timestamp (non requis)

        retourne la position dans la timeserie Text et le timestamp correspondant
        """
        if ts is None:
            start = self._tss
            tse = self._tse
            end = tse - self._wsize * self._interval - 4*24*3600
            #print(tsToHuman(start),tsToHuman(end))
            # on tire un timestamp avant fin mai OU après début octobre
            ts = getRandomStart(start, end, 10, 5)
        self._pos = (ts - self._tss) // self._interval
        self._tsvrai = self._tss + self._pos * self._interval

        print("*************************************")
        print("{} - {}".format(ts,tsToHuman(ts)))
        print("vrai={} - {}".format(self._tsvrai, tsToHuman(self._tsvrai)))

    def buildEnv(self, Tint=None):
        """
        retourne le tenseur des données de l'épisode

        Tint : valeur initiale de température intérieure
        Fournir un entier pour Tint permet de fixer la température intérieure du premier point de l'épisode
        Si Tint vaut None, un tirage aléatoire entre 17 et 20 est réalisé

        caractéristiques du tenseur de sortie

        - axe 0 = le temps
        - axe 1 = les paramètres pour décrire l'environnement

        3 paramètres physiques : Qc, Text et Tint

        2 paramètres organisationnels :

        - temperature de consigne * occupation - si > 0 : bâtiment occupé,
        - nombre d'heures d'ici le changement d 'occupation
        """
        datas=np.zeros((self._wsize, 5))
        # condition initiale en température
        if isinstance(Tint, (int, float)):
            datas[0,2] = Tint
        else:
            datas[0,2] = random.randint(17,20)
        # on connait Text (vérité terrain) sur toute la longueur de l'épisode
        datas[:,1] = self._Text[self._pos:self._pos+self._wsize]
        occupation = self._agenda[self._pos:self._pos+self._wsize+4*24*3600//self._interval]
        for i in range(self._wsize):
            datas[i,4] = getLevelDuration(occupation, i) * self._interval / 3600
        # consigne
        datas[:,3] = self._Tc * occupation[0:self._wsize]
        print("condition initiale : Text {:.2f} Tint {:.2f}".format(datas[0,1],datas[0,2]))
        return datas

    def sim(self, datas, i):
        """
        calcule la température à l'étape i
        """
        _Qc = datas[i-1:i+1,0]
        _Text = datas[i-1:i+1,1]
        return R1C1(self._interval, self._model["R"], self._model["C"], _Qc, _Text, datas[i-1,2], cte=self._cte)

    def sim2Target(self, datas, i):
        """
        on est à l'étape i et on veut calculer la température à l'ouverture des locaux, en chauffant dès à présent en permanence
        """
        # pour pouvoir se balader dans les vecteurs et calculer à la cible, on repasse en nombre d'intervalles
        tof = int(datas[i, 4] * 3600 / self._interval)
        # ON VEUT CONNAITRE LA TEMPERATURE INTERIEURE A self._pos+i+tof
        Qc = np.ones(tof+1)*self._max_power
        # datas[i,1] correspond à Text[i+pos]
        Text = self._Text[self._pos+i:self._pos+i+tof+1]
        Tint = datas[i, 2]
        return R1C1sim(self._interval, self._model["R"], self._model["C"], Qc, Text, Tint, cte=self._cte)

    def play(self, datas):
        """
        à définir dans la classe fille pour jouer une stratégie de chauffe

        retourne le tenseur de données sources complété par le scénario de chauffage et la température intérieure simulée
        """
        return datas

class Evaluate:
    """
    name : nom qui sera utilisé par la fonction close pour sauver le réseau et les graphiques

    name est compris par close comme un chemin et les graphes sont tjrs enregistrés dans le répertoire contenant l'agent

    """
    def __init__(self, name, env, agent, **params):
        self._N = MAX_EPISODES
        if "N" in params:
            self._N = params["N"]
        self._k = params["k"] if "k" in params else 1

        print("on va jouer {} épisodes".format(self._N))
        self._name = name
        self._env = env
        self._modlabel = "R={:.2e} C={:.2e}".format(self._env._model["R"], self._env._model["C"])
        self._agent = agent
        print("métrique de l'agent online {}".format(agent.metrics_names))
        self._LNames, self._inSize, self._outSize = getConfig(agent)
        self._exit = False
        # sert uniquement pour évaluer la durée de l'entrainement
        self._ts = int(time.time())
        # numéro de l'épisode
        self._steps = 0
        self._policy = "agent"
        """
        _rewards sert à enregistrer le détail de la structure de la récompense sur un épisode
        exemple : partie confort, partie vote, partie energy
        à mettre à jour dans la méthode reward() qui est à définir dans la classe fille
        """
        from collections import defaultdict
        ini = defaultdict(lambda:np.zeros(self._env._wsize))
        self._rewards = {"agent":ini, "model":copy.deepcopy(ini)}
        """
        on parle de luxe si la température intérieure est supérieure à Tc+hh
        on parle d'inconfort si la température intérieure est inférieure à Tc-hh
        les colonnes de la matrice stats sont les suivantes :
        - 0 : timestamp de l'épisode,
        - 1 à 4 : agent température intérieure moyenne, nb pts luxe, nb pts inconfort, consommation
        - 5 à 8 : modèle idem
        - 9 à 10 : récompense agent puis modèle
        """
        self._stats = np.zeros((self._N, 11))
        self._multiAgent = False

    def setOccupancyAgent(self, agent):
        self._multiAgent = True
        self._occupancyAgent = agent

    def _sigint_handler(self, signal, frame):
        print("signal de fermeture reçu")
        self._exit = True

    def stats(self, datas):
        w = datas[datas[:,3]!=0,2]
        inc = w[w[:] < self._env._Tc - self._env._hh]
        luxe = w[w[:] > self._env._Tc + self._env._hh]
        Tocc_moy = round(np.mean(w[:]),2)
        nbinc = inc.shape[0] * self._env._interval // 3600
        nbluxe = luxe.shape[0] * self._env._interval // 3600
        return Tocc_moy, nbinc, nbluxe

    def play(self, silent, ts=None, snapshot=False, Tint = None):
        """
        joue un épisode

        retourne le tenseur des données de l'agent, au cas où on souhaite y vérifier un détail

        ts : int - timestamp que l'on veut rejouer, si None, un tirage alétaoire est réalisé

        snapshot : boolean - si True, l'image n'est pas affichée et un fichier tiers utilisant la classe peut l'enregistrer

        Tint : condition initiale de température intérieure
        """
        self._env.setStart(ts)
        adatas = self._env.buildEnv(Tint=Tint)
        wsize = adatas.shape[0]

        mdatas = self._env.play(copy.deepcopy(adatas))
        mConso = int(np.sum(mdatas[1:,0]) / 1000) * self._env._interval // 3600
        self._rewards["agent"].clear()
        self._rewards["model"].clear()
        cumulArewards = []
        cumulMrewards = []
        ar = 0
        mr = 0

        for i in range(1,wsize):
            if self._LNames == ['states', 'dense', 'dense_1']:
                if self._inSize == 5 :
                    state = np.zeros((self._inSize))
                    state[0] = adatas[i-1, 2]
                    state[1] = adatas[i-1, 1]
                    state[2] = 1 if adatas[i-1, 3] > 0 else 0
                    state[3] = adatas[i-1, 4]
                    state[4] = adatas[i-1, 3]
                else:
                    # on permute Tint et Text car les agents jusque début 2021 prenaient Tint en premier....
                    # on pourrait utiliser np.array([ adatas[i-1,2], adatas[i-1,1], adatas[i-1,3], adatas[i-1,4] ])
                    # mais le slicing donne un code plus lisible et plus court :-)
                    reorder = [2,1,3,4]
                    state = adatas[i-1, reorder[0:self._inSize]]
            else:
                state = adatas[i-1, 1:self._inSize + 1]
            agent = self._agent
            if self._multiAgent:
                agent = self._occupancyAgent if state[2] != 0 else self._agent
            predictionBrute = agent(state.reshape(1, self._inSize))
            action = np.argmax(predictionBrute)
            adatas[i-1,0] = action * self._env._max_power
            # on peut désormais calculer la récompense à l'étape i-1
            self._policy = "agent"
            ar += self.reward(adatas, i-1)
            self._policy = "model"
            mr += self.reward(mdatas, i-1)
            cumulArewards.append(ar)
            cumulMrewards.append(mr)
            # calcul de la température à l'état suivant
            adatas[i,2] = self._env.sim(adatas, i)
        aConso = int(np.sum(adatas[1:,0]) / 1000) * self._env._interval // 3600
        print("récompense agent {:.2f} récompense modèle {:.2f}".format(ar,mr))

        # on ne prend pas le premier point de température intérieure car c'est une condition initiale arbitraire
        aTocc_moy, aNbinc, aNbluxe = self.stats(adatas[1:,:])
        mTocc_moy, mNbinc, mNbluxe = self.stats(mdatas[1:,:])
        line = np.array([self._env._tsvrai, aTocc_moy, aNbluxe, aNbinc, aConso, mTocc_moy, mNbluxe, mNbinc, mConso, ar, mr])
        #print(line)
        self._stats[self._steps, :] = line

        if not silent or snapshot:
            Tmin = min( np.min(mdatas[:,2]) , np.min(adatas[:,2]) )
            Tmax = max( np.max(mdatas[:,2]) , np.min(adatas[:,2]) )
            Tc = self._env._Tc
            hh = self._env._hh
            tsvrai = self._env._tsvrai
            pos = self._env._pos
            interval = self._env._interval
            occupation = self._env._agenda[pos:pos + wsize + 4*24*3600 // interval]
            xr, zoneconfort, zonesOcc = covering(Tmin, Tmax, Tc, hh, tsvrai, wsize, interval, occupation)

            title = "épisode {} - {} {} {}".format(self._steps, tsvrai, tsToHuman(tsvrai), self._modlabel)
            title = "{}\n conso Modèle {} Agent {}".format(title, mConso, aConso)
            title = "{}\n Tocc moyenne modèle : {} agent : {}".format(title, mTocc_moy, aTocc_moy)
            title = "{}\n nb heures inconfort modèle : {} agent : {}".format(title, mNbinc, aNbinc)

            if snapshot:
                plt.figure(figsize=(20, 10))

            nbg = 411
            ax1 = plt.subplot(nbg)
            plt.title(title, fontsize=8)
            ax1.add_patch(zoneconfort)
            for v in zonesOcc:
                ax1.add_patch(v)
            plt.ylabel("Temp. intérieure °C")
            plt.plot(xr, mdatas[:,2], color="orange", label="TintMod")
            plt.plot(xr, adatas[:,2], color="black", label="TintAgent")
            plt.legend(loc='upper left')

            ax2 = ax1.twinx()
            plt.ylabel("Temp. extérieure °C")
            plt.plot(xr, mdatas[:,1], color="blue", label="Text")
            plt.legend(loc='upper right')

            nbg+=1
            ax3 = plt.subplot(nbg, sharex=ax1)
            plt.ylabel("Conso W agent")
            plt.plot(xr, adatas[:,0], color="black", label="consoAgent")
            plt.legend(loc='upper left')

            ax4 = ax3.twinx()
            plt.ylabel("cum.reward agent")
            a = "agent"
            y = np.zeros(wsize)
            for r in self._rewards[a]:
                ax4.fill_between(xr, y, y+self._rewards[a][r], alpha=0.6, label="{} {}".format(a,r))
                y = y+self._rewards[a][r]
            #plt.plot(xr[1:], cumulArewards, color="black", label="agent")
            plt.legend(loc='upper right')

            nbg+=1
            ax5 = plt.subplot(nbg, sharex=ax1)
            plt.ylabel("Conso W modèle")
            plt.plot(xr, mdatas[:,0], color="orange", label="consoMod")
            plt.legend(loc='upper left')

            ax6 = ax5.twinx()
            plt.ylabel("cum.reward mod")
            a = "model"
            y = np.zeros(wsize)
            for r in self._rewards[a]:
                ax6.fill_between(xr, y, y+self._rewards[a][r], alpha=0.6, label="{} {}".format(a,r))
                y = y+self._rewards[a][r]
            #plt.plot(xr[1:], cumulMrewards, color="orange", label="mod")
            plt.legend(loc='upper right')

            # à enlever si on veut alléger le graphique
            nbg+=1
            ax7 = plt.subplot(nbg, sharex=ax1)
            plt.ylabel("°C")
            plt.plot(xr, mdatas[:,3], label="consigne")
            plt.legend(loc='upper left')
            ax8 = ax7.twinx()
            plt.ylabel("nb steps > cgt occ.")
            plt.plot(xr, mdatas[:,4],'o', markersize=1, color="red")

            if not snapshot:
                plt.show()
        return adatas

    def reward(self, datas, i):
        """
        fonction récompense

        à définir dans la classe fille
        """
        return 0

    def run(self, silent=False, Tint=None):
        """
        boucle d'exécution
        """
        signal.signal(signal.SIGINT, self._sigint_handler)
        signal.signal(signal.SIGTERM, self._sigint_handler)

        while not self._exit:
            if self._steps >= self._N - 1:
                self._exit = True

            self.play(silent=silent, Tint=Tint)
            self._steps += 1

            time.sleep(0.1)

    def close(self, suffix=None):
        """
        enregistre les statistiques (csv + png) si on est arrivé au bout du nombre d'épisodes

        suffix, s'il est fourni, sert dans la construction de(s) nom(s) de fichier(s),
        pour préciser par ex. le type de politique optimale jouée par le modèle

        """

        statsMoy = np.mean(self._stats, axis = 0).round(1)

        print("leaving the game")
        # enregistrement des statistiques du jeu
        # uniquement si on est allé au bout des épisodes - pas la peine de sauver des figures vides
        # on utilise le suffixe pour indiquer le mode de jeu du modèle
        if self._steps == self._N :

            title = "modèle {} jouant la politique optimale {}\n".format(self._modlabel, suffix) if suffix is not None else ""
            title = "{} Conso moyenne agent : {} / Conso moyenne modèle : {}\n".format(title, statsMoy[4], statsMoy[8])

            pct = round(100*(statsMoy[8]-statsMoy[4])/statsMoy[8], 2)
            title = "{} Pourcentage de gain agent : {} %".format(title, pct)

            plt.figure(figsize=(20, 10))
            ax1 = plt.subplot(411)
            plt.title(title)
            plt.plot(self._stats[:,1], color="blue", label='température moyenne occupation agent')
            plt.plot(self._stats[:,5], color="red", label='température moyenne occupation modèle')
            plt.legend()

            ax2 = plt.subplot(412, sharex=ax1)
            plt.plot(self._stats[:,2], color="blue", label="nombre heures > {}°C agent".format(self._env._Tc + self._env._hh))
            plt.plot(self._stats[:,6], color="red", label="nombre heures > {}°C modèle".format(self._env._Tc + self._env._hh))
            plt.legend()

            ax3 = plt.subplot(413, sharex=ax1)
            plt.plot(self._stats[:,3], color="blue", label="nombre heures < {}°C agent".format(self._env._Tc - self._env._hh))
            plt.plot(self._stats[:,7], color="red", label="nombre heures < {}°C modèle".format(self._env._Tc - self._env._hh))
            plt.legend()

            ax4 = plt.subplot(414, sharex=ax1)
            plt.plot(self._stats[:,9], color="blue", label="récompense cumulée agent")
            plt.plot(self._stats[:,10], color="red", label="récompense cumulée modèle")
            plt.legend()

            ts = time.time()
            now = tsToHuman(ts, fmt="%Y_%m_%d_%H_%M")
            label = "played_{}".format(suffix) if suffix is not None else "played"

            name = "{}_{}_{}".format(self._name.replace(".h5",""),label,now)
            plt.savefig(name)
            header = "ts"
            for w in ["agent","modèle"]:
                header = "{0},{1}_Tintmoy,{1}_nbpts_luxe,{1}_nbpts_inconfort,{1}_conso".format(header,w)
            np.savetxt('{}.csv'.format(name), self._stats, delimiter=',', header=header)

        plt.close()

        return statsMoy
