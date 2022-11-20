"""heatgym module"""
import random
from typing import Optional
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

def covering(tmin, tmax, tc, hh, ts, wsize, interval, occupation, xr=None):
    """
    permet l'habillage du graphique d'un épisode
    avec la zone de confort et les périodes d'occupation
    utilise la fonction Rectangle de matplotlib

    - tmin : température minimale
    - tmax : température maximale
    - tc : température de consigne
    - hh : demi-intervalle de la zone de confort
    - nombre de points dans l'épisode
    - interval : pas de temps
    - occupation : agenda d'occupation avec 4 jours supplémentaires pour calculer les durées d'occupation

    retourne les objets matérialisant les zones de confort et d'occupation
    """
    if xr is None:
        xrs = np.arange(ts, ts + wsize * interval, interval)
        xr = np.array(xrs, dtype='datetime64[s]')

    zone_confort = Rectangle((xr[0], tc-hh), xr[-1]-xr[0], 2 * hh,
                             facecolor='g', alpha=0.5, edgecolor='None',
                             label="zone de confort")
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

    return xr, zone_confort, zones_occ


class Vacancy(gym.Env):
    """mode hors occupation"""
    def __init__(self, text, max_power, tc, k, **model):
        """
        text : objet PyFina de température extérieure

        state : tableau numpy de 4 éléments :
        - température extérieure
        - température intérieure
        - tc * occupation
        - nb hours -> occupation change (from occupied to empty and vice versa)
        """
        super().__init__()
        self.text = text
        self._tss = text.start
        self._tse = text.start + text.step * text.shape[0]
        # pas de temps en secondes
        self._interval = text.step
        # nombre de pas dans un épisode (taille de la fenêtre)
        self.wsize = None
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
        self._max_power = max_power
        self._tc = tc
        self._k = k
        self.model = model if model else MODELRC
        # la constante de temps du modèle électrique équivalent
        self._tcte = self.model["R"] * self.model["C"]
        self._cte = math.exp(-self._interval/self._tcte)
        self.action_space = spaces.Discrete(2)
        high = np.finfo(np.float32).max
        self.observation_space = spaces.Box(-high, high, (4,), dtype=np.float32)
        #print(self.observation_space)
        self.state = None
        # labels
        self.label = "vacancy"
        self.reward_label = None
        # paramètres pour le rendu graphique
        self._fig = None
        self._ax1 = None
        self._ax2 = None
        self._ax3 = None


    def _reset(self, ts=None, tint=None, seed:Optional[int] = None):
        """
        generic reset method

        Avant d'appeler cette méthode, il faut fixer la taille de l'épisode wsize !

        permet de définir self._xr, self.pos, self.tsvrai
        ces grandeurs sont des constantes de l'épisode

        initialise self.i, self._tot_reward, self.tint, self.action
        ces grandeurs sont mises à jour à chaque pas de temps

        retourne self.state
        """
        self.seed(seed)
        if ts is None:
            start = self._tss
            tse = self._tse
            end = tse - self.wsize * self._interval - 4*24*3600
            # on tire un timestamp avant fin mai OU après début octobre
            ts = get_random_start(start, end, 10, 5)
        self.i = 0
        self.pos = (ts - self._tss) // self._interval
        self.tsvrai = self._tss + self.pos * self._interval
        #print("episode timestamp : {}".format(self.tsvrai))
        # x axis = time for human
        xrs = np.arange(self.tsvrai, self.tsvrai + self.wsize * self._interval, self._interval)
        self._xr = np.array(xrs, dtype='datetime64[s]')
        text = self.text[self.pos]
        if not isinstance(tint, (int, float)):
            tint = random.randint(17, 20)
        self.tint = np.zeros(self.wsize + 1)
        self.action = np.zeros(self.wsize + 1)
        self.tint[self.i] = tint
        tc, nbh = self.update_non_phys_params()
        self.state = np.array([text, tint, tc, nbh], dtype=np.float32)
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
        title = f'{self.tsvrai} - score : {self._tot_reward:.2f}'
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
        q_c = action * self._max_power
        self.action[self.i] = action
        # indoor temps at next state
        text = self.text[self.pos+self.i:self.pos+self.i+2]
        delta = self._cte * (q_c / self.model["C"] + text[0] / self._tcte)
        delta += q_c / self.model["C"] + text[1] / self._tcte
        self.i += 1
        x = self.state[1] * self._cte + self._interval * 0.5 * delta
        self.tint[self.i] = x
        # non physical parameters at next state
        tc, nbh = self.update_non_phys_params()
        # on met à jour state avec les données de next state
        self.state = np.array([text[1], x, tc, nbh], dtype=np.float32)
        # return reward at state
        return reward

    def update_non_phys_params(self):
        """
        estimates :
        - tc*occupation
        - nbh -> occupation change (from occupied to empty and vice versa)

        *Note that 1h30 has to be coded as 1.5*

        In the vacancy case, tc*occupation is always 0
        """
        return 0, (self.wsize - 1 - self.i) * self._interval / 3600

    def reward(self, action):
        """reward at state action"""
        self.reward_label = "Vote_final_reward_only"
        reward = 0
        tc = self._tc
        if self.state[3] == 0 :
            # l'occupation du bâtiment commence
            # pour converger vers la température cible
            if self.state[1] >= tc :
                reward = -15 * (self.state[1] - tc - 1) - 20
            if self.state[1] < tc :
                reward = -15 * (tc - 1 - self.state[1]) - 20
            # le bonus énergétique
            if self.state[1] <= tc + 1 and self.state[1] >= tc - 3:
                reward += self.tot_eko * self._k * self._interval / 3600
        # calcul de l'énergie économisée
        if not action :
            self.tot_eko += 1
        return reward

    def reset(self, ts=None, tint=None, seed:Optional[int] = None, wsize=None):
        """episode reset"""
        if not isinstance(wsize, int):
            self.wsize = 63 * 3600 // self._interval
        else :
            self.wsize = wsize
        self.tot_eko = 0
        if not isinstance(tint, (int, float)):
            tint = 20
        return self._reset(ts=ts, tint=tint, seed=seed)

    def step(self, action):
        """return state, reward pour previous state, done, _"""
        reward = self._step(action)
        done = True if self.i == self.wsize else None
        return self.state, reward, done, {}

    def render(self, stepbystep=True, label=None):
        """render realtime or not"""
        self._render(stepbystep=stepbystep, label = label)

    def close(self):
        """closing"""
        #plt.savefig("test.png")
        plt.close()

class Building(Vacancy):
    """mode universel - alternance d'occupation et de non-occupation"""
    def __init__(self, text, agenda, wsize, max_power, tc, k, **model):
        super().__init__(text, max_power, tc, k, **model)
        self._agenda = agenda
        self.wsize = wsize
        self.label = "week"

    def update_non_phys_params(self):
        pos1 = self.pos + self.i
        tc = self._agenda[pos1] * self._tc
        pos2 = self.pos + self.i + self.wsize + 4 * 24 * 3600 // self._interval
        occupation = self._agenda[pos1:pos2]
        nbh = get_level_duration(occupation, 0) * self._interval / 3600
        return tc, nbh

    def reward(self, action):
        reward = 0
        tc = self._tc
        if self.state[2] != 0:
            l_0 = tc - 5
            l_1 = tc - 3
            l_2 = tc - 1
            if abs(self.state[1] - tc) > 1:
                reward -= abs(self.state[1] - tc) * self._interval / 3600
            if self._agenda[self.pos+self.i-1] == 0:
                if self.state[1] < l_0:
                    reward -= 30
                if self.state[1] < l_1:
                    reward -= 30
                if self.state[1] < l_2:
                    reward -= 20
        else:
            if action :
                reward -= (self._k + max(0, self.state[1] - tc)) * self._interval / 3600
            else :
                self.tot_eko += 1
        return reward

    def reset(self, ts=None, tint=None, seed: Optional[int] = None, wsize=None):
        self.tot_eko = 0
        return self._reset(ts=ts, tint=tint, seed=seed)

    def step(self, action):
        reward = self._step(action)
        done = None
        if self.i == self.wsize - 1:
            done = True
        return self.state, reward, done, {}

    def render(self, stepbystep=True, label=None):
        if self.i:
            tmin = np.min(self.tint[0: self.i])
            tmax = np.max(self.tint[0: self.i])
            occupation = self._agenda[self.pos:self.pos+self.wsize+4*24*3600//self._interval]
            _, zone_confort, zones_occ = covering(tmin, tmax, self._tc, 1,
                                                 self.tsvrai, self.wsize, self._interval,
                                                 occupation, self._xr)
            self._render(zone_confort=zone_confort, zones_occ=zones_occ, stepbystep=stepbystep)
        else:
            self._render(stepbystep=stepbystep)