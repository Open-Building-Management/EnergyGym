"""heatgym"""
import random
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import gym
from gym import spaces
import numpy as np

from .planning import get_random_start, get_level_duration

# modèle par défault de type R1C1 obtenues par EDW avec les données de Marc Bloch
#MODELRC = {"R": 2.54061406e-04, "C": 9.01650468e+08}
MODELRC = {"R": 5.94419964e-04, "C": 5.40132642e+07}

def confort(xr, tc, hh):
    """construit le rectangle vert de la zone de confort thermique
    facecolor='g': green color
    """
    return Rectangle((xr[0], tc-hh), xr[-1]-xr[0], 2 * hh,
                     facecolor='g', alpha=0.5, edgecolor='None',
                     label="zone de confort")


def presence(xr, occupation, wsize, tmin, tmax, tc, hh):
    """construit les rectangles orange matérialisant l'occupation"""
    changes = []
    for i in range(wsize):
        if occupation[i] == 0 and occupation[i+1] != 0:
            changes.append(i)
    zones_occ = []
    hauteur = max(tc + hh, tmax) - min(tc - hh, tmin)
    for i in changes:
        if i < wsize-1:
            largeur = get_level_duration(occupation, i+1) * (xr[1] - xr[0])
            occ = Rectangle((xr[i], tmin), largeur, hauteur,
                            facecolor='orange', alpha=0.5, edgecolor='None')
            zones_occ.append(occ)
    return zones_occ


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

class Env(gym.Env):
    """base environnement"""
    def __init__(self, text, max_power, tc, k, **model):
        """
        text : objet PyFina de température extérieure
        """
        super().__init__()
        self.text = text
        self._tss = text.start
        self._tse = text.start + text.step * text.shape[0]
        # pas de temps en secondes
        self._interval = text.step
        # nombre de pas dans un épisode (taille de la fenêtre)
        self.wsize = None
        # nombre de pas que l'on peut remonter dans l'histoire passée
        pastsize = model.get("pastsize", 1)
        if "nbh" in model:
            pastsize = model["nbh"] * 3600 // self._interval
        self.pastsize = int(pastsize)
        # tableau numpy des températures passées
        self.tint_past = np.zeros(self.pastsize)
        self.text_past = np.zeros(self.pastsize)
        self.q_c_past = np.zeros(self.pastsize - 1)
        # timestamp du début de l'épisode
        self.tsvrai = None
        # position de l'épisode dans la timesérie text
        self.pos = None
        # compteur de pas dans l'épisode
        self.i = 0
        # échelle de temps de l'épisode au format humain
        self._xr = None
        # récompense accumulée au cours de l'épisode
        self._tot_reward = None
        # tableau numpy des températures intérieures au cours de l'épisode
        # de taille wsize + 1
        self.tint = None
        # tableau numpy des actions énergétiques au cours de l'épisode
        # de taille wsize + 1
        self.action = None
        # nombre de pas de temps sans conso énergétique pour l'épisode
        self.tot_eko = 0
        self.max_power = max_power
        self._tc = tc
        self.tc_episode = None
        self._k = k
        self.model = model if model else MODELRC
        # la constante de temps du modèle électrique équivalent
        self.tcte = self.model["R"] * self.model["C"]
        self.cte = math.exp(-self._interval/self.tcte)
        # current state in the observation space
        self.state = None
        # labels
        self.reward_label = None
        # paramètres pour le rendu graphique
        self._fig = None
        self._ax1 = None
        self._ax2 = None
        self._ax3 = None
        self.agenda = None

    def _reset(self, ts=None, tint=None, tc_episode=None):
        """
        generic reset method

        Avant d'appeler cette méthode, il faut fixer la taille de l'épisode wsize !

        permet de définir self._xr, self.pos, self.tsvrai
        ces grandeurs sont des constantes de l'épisode

        initialise self.i, self._tot_reward, self.tint, self.action
        ces grandeurs sont mises à jour à chaque pas de temps

        retourne self.state
        """
        if ts is None:
            start = self._tss + self.pastsize * self._interval
            tse = self._tse
            end = tse - self.wsize * self._interval - 4*24*3600
            # on tire un timestamp avant fin mai OU après début octobre
            ts = get_random_start(start, end, 10, 5)
        self.i = 0
        self.pos = (ts - self._tss) // self._interval
        self.tsvrai = self._tss + self.pos * self._interval
        #print("episode timestamp : {}".format(self.tsvrai))
        # on fixe la température de consigne (à la cible) de notre épisode
        if isinstance(tc_episode, (int, float)):
            self.tc_episode = tc_episode
        else:
            self.tc_episode = self._tc + random.randint(-2,2)
        # x axis = time for human
        xrs = np.arange(self.tsvrai, self.tsvrai + self.wsize * self._interval, self._interval)
        self._xr = np.array(xrs, dtype='datetime64[s]')
        self.tint = np.zeros(self.wsize + 1)
        self.action = np.zeros(self.wsize + 1)
        # construction d'une histoire passée
        if not isinstance(tint, (int, float)):
            tint = self.tc_episode + random.randint(-3, 0)
        self.tint_past[0] = tint
        action = self.tint_past[0] <= self.tc_episode
        q_c = action * self.max_power
        self.text_past = self.text[self.pos - self.pastsize + 1: self.pos + 1]
        for i in range(1, self.pastsize):
            self.q_c_past[i-1] = q_c
            pos = self.pos - self.pastsize + 1 + i
            delta = self.cte * (q_c / self.model["C"] + self.text[pos-1] / self.tcte)
            delta += q_c / self.model["C"] + self.text[pos] / self.tcte
            self.tint_past[i] = self.tint_past[i-1] * self.cte + self._interval * 0.5 * delta
            if self.tint_past[i] >= self.tc_episode + 1 or self.tint_past[i] <= self.tc_episode - 1:
                action = self.tint_past[i] <= self.tc_episode
            q_c = action * self.max_power
        # on vient de s'arrêter à self.pos
        # on a donc notre condition initiale en température intérieure
        self.tint[0] = self.tint_past[-1]
        # construction de state
        self.state = self._state()
        self._tot_reward = 0
        return self.state

    def _render(self, zone_confort=None, zones_occ=None, stepbystep=True, label=None):
        """generic render method"""
        if self.i == 0 or not stepbystep:
            self._fig = plt.figure()
            self._ax1 = plt.subplot(311)
            self._ax2 = plt.subplot(312, sharex=self._ax1)
            self._ax3 = plt.subplot(313, sharex=self._ax1)
            if stepbystep :
                plt.ion()
        title = f'{self.tsvrai} - score: {self._tot_reward:.2f} - tc_episode: {self.tc_episode}°C'
        if label is not None :
            title = f'{title}\n{label}'
        self._fig.suptitle(title)
        self._ax1.clear()
        self._ax2.clear()
        self._ax3.clear()
        self._ax1.plot(self._xr, self.text[self.pos:self.pos+self.wsize])
        self._ax2.plot(self._xr[0:self.i], self.tint[0:self.i])
        self._ax3.plot(self._xr[0:self.i], self.action[0:self.i])
        if self.i :
            if zone_confort is not None :
                self._ax2.add_patch(zone_confort)
            if zones_occ is not None :
                for occ in zones_occ:
                    self._ax2.add_patch(occ)
        plt.show()
        if stepbystep:
            plt.pause(0.0001)

    def _step(self, action):
        """generic step method"""
        err_msg = f'{action} is not a valid action'
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."
        # reward at state
        reward = self.reward(action)
        self._tot_reward += reward
        # Qc at state
        q_c = action * self.max_power
        self.action[self.i] = action
        # indoor temp at next state
        text = self.text[self.pos+self.i:self.pos+self.i+2]
        delta = self.cte * (q_c / self.model["C"] + text[0] / self.tcte)
        delta += q_c / self.model["C"] + text[1] / self.tcte
        tint = self.tint[self.i] * self.cte + self._interval * 0.5 * delta
        self.i += 1
        self.tint[self.i] = tint
        # on met à jour state avec les données de next state
        self.tint_past = np.array([*self.tint_past[1:], tint])
        self.text_past = np.array([*self.text_past[1:], text[1]])
        if self.pastsize > 1:
            self.q_c_past = np.array([*self.q_c_past[1:], q_c])
        self.state = self._state()
        # return reward at state
        return reward

    def _state(self):
        """return the current state after all calculations are done
        example de state
        pour une température de consigne maintenue constante sur l'épisode
        à surcharger dans classe fille"""
        tc = self.tc_episode
        return np.array([*self.text_past,
                         *self.tint_past,
                         *self.q_c_past,
                         tc], dtype=np.float32)

    def _covering(self):
        """retourne la zone de confort
        et les éventuelles zones d'occupation si un agenda est en place"""
        zone_confort = confort(self._xr, self.tc_episode, 1)
        zones_occ = None
        if self.agenda is not None:
            tmin = np.min(self.tint[0: self.i])
            tmax = np.max(self.tint[0: self.i])
            occupation = self.agenda[self.pos:self.pos+self.wsize+4*24*3600//self._interval]
            zones_occ = presence(self._xr, occupation, self.wsize, tmin, tmax, self.tc_episode, 1)
        return zone_confort, zones_occ

    def set_agenda(self, agenda):
        """définit l'agenda d'occupation"""
        self.agenda = agenda

    def reward(self, action):
        """récompense hystéresis
        à surcharger dans classe fille"""
        reward = 0
        tc = self.tc_episode
        tint = self.tint[self.i]
        reward = - abs(tint - tc) * self._interval / 3600
        # calcul de l'énergie économisée
        if not action :
            self.tot_eko += 1
        return reward

    def reset(self, ts=None, tint=None, tc_episode=None, wsize=None):  # pylint: disable=W0221
        """episode reset"""
        if not isinstance(wsize, int):
            self.wsize = 63 * 3600 // self._interval
        else :
            self.wsize = int(wsize)
        self.tot_eko = 0
        return self._reset(ts=ts, tint=tint, tc_episode=tc_episode)

    def step(self, action):
        """return state, reward pour previous state, done, _"""
        reward = self._step(action)
        done = True if self.i == self.wsize else None
        return self.state, reward, done, {}

    def render(self, stepbystep=True, label=None):
        """render realtime or not"""
        self._render(stepbystep=stepbystep, label=label)

    def close(self):
        """closing"""
        #plt.savefig("test.png")
        plt.close()

class Hyst(Env):
    """mode hystéresis permanent"""
    def __init__(self, text, max_power, tc, k, **model):
        """
        state : tableau numpy :
        - historique de température extérieure de taille n
        - historique de température intérieure de taille n
        - historique de chauffage de taille n-1
        - tc, consigne de température intérieure
        avec n=1, l'espace d'observation est de taille 3
        """
        super().__init__(text, max_power, tc, k, **model)
        self.action_space = spaces.Discrete(2)
        high = np.finfo(np.float32).max
        self.observation_space = spaces.Box(-high, high, (3*self.pastsize,), dtype=np.float32)
        # labels
        self.label = "hysteresis"
        self.reward_label = "hysteresis"

    def render(self, stepbystep=True, label=None):
        """avec affichage de la zone de confort"""
        if self.i:
            zone_confort, zones_occ = self._covering()
            self._render(zone_confort=zone_confort, zones_occ=zones_occ, stepbystep=stepbystep, label=label)
        else:
            self._render(stepbystep=stepbystep, label=label)


class Vacancy(Env):
    """mode hors occupation"""
    def __init__(self, text, max_power, tc, k, **model):
        """
        state : tableau numpy :
        - historique de température extérieure de taille n
        - historique de température intérieure de taille n
        - historique de chauffage de taille n-1
        - température de consigne à la cible
        - nb hours -> occupation change (from occupied to empty and vice versa)
        """
        super().__init__(text, max_power, tc, k, **model)
        self.action_space = spaces.Discrete(2)
        high = np.finfo(np.float32).max
        self.observation_space = spaces.Box(-high, high, (3*self.pastsize+1,), dtype=np.float32)
        #print(self.observation_space)
        # labels
        self.label = "vacancy"

    def _state(self):
        """return the current state after all calculations are done"""
        tc = self.tc_episode
        # nbh -> occupation change (from occupied to empty and vice versa)
        # Note that 1h30 has to be coded as 1.5
        nbh = (self.wsize - 1 - self.i) * self._interval / 3600
        return np.array([*self.text_past,
                         *self.tint_past,
                         *self.q_c_past,
                         tc,
                         nbh], dtype=np.float32)

    def reward(self, action):
        """reward at state action"""
        self.reward_label = "Vote_final_reward_only"
        reward = 0
        tc = self.tc_episode
        tint = self.tint[self.i]
        if self.state[-1] == 0 :
            # l'occupation du bâtiment commence
            # pour converger vers la température cible
            reward = - 15 * abs(tint - tc) * self._interval / 3600
            # le bonus énergétique
            if tc - 3 <= tint <= tc + 1 :
                reward += self.tot_eko * self._k * self._interval / 3600
        if not action :
            self.tot_eko += 1
        return reward


class Building(Vacancy):
    """mode universel - alternance d'occupation et de non-occupation
    needed for tests, not really for trainings
    """
    def __init__(self, text, agenda, max_power, tc, k, **model):
        super().__init__(text, max_power, tc, k, **model)
        self.agenda = agenda
        self.label = "week"

    def _state(self):
        """return the current state after all calculations are done"""
        pos1 = self.pos + self.i
        tc = self.agenda[pos1] * self.tc_episode
        pos2 = self.pos + self.i + self.wsize + 4 * 24 * 3600 // self._interval
        occupation = self.agenda[pos1:pos2]
        nbh = get_level_duration(occupation, 0) * self._interval / 3600
        return np.array([*self.text_past,
                         *self.tint_past,
                         *self.q_c_past,
                         tc,
                         nbh], dtype=np.float32)

    def reward(self, action):
        reward = 0
        if self.state[-2] != 0:
            tc = self.state[-2]
            tint = self.tint[self.i]
            reward -= abs(tint - tc) * self._interval / 3600
            if self.agenda[self.pos+self.i-1] == 0:
                # was the building empty at previous state ?
                if tc - 3 <= tint <= tc + 1 :
                    reward += self.tot_eko * self._k * self._interval / 3600
            else :
                if self.tot_eko:
                    self.tot_eko = 0
        else:
            if not action :
                self.tot_eko += 1
        return reward

    def reset(self, ts=None, tint=None, tc_episode=None, wsize=None):
        if not isinstance(wsize, int):
            self.wsize = 1 + 8*24*3600 // self._interval
        else :
            self.wsize = int(wsize)
        self.tot_eko = 0
        return self._reset(ts=ts, tint=tint, tc_episode=tc_episode)

    def step(self, action):
        reward = self._step(action)
        done = True if self.i == self.wsize - 1 else None
        return self.state, reward, done, {}

    def render(self, stepbystep=True, label=None):
        if self.i:
            zone_confort, zones_occ = self._covering()
            self._render(zone_confort=zone_confort, zones_occ=zones_occ, stepbystep=stepbystep)
        else:
            self._render(stepbystep=stepbystep)
