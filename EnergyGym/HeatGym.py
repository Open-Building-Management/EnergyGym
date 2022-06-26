import gym
from gym import spaces
import numpy as np
import random
from typing import Optional
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .planning import getRandomStart, getLevelDuration

# modèle par défault de type R1C1 obtenues par EDW avec les données de Marc Bloch
#modelRC = {"R": 2.54061406e-04, "C": 9.01650468e+08}
modelRC = {"R" : 5.94419964e-04, "C" : 5.40132642e+07}

def covering(Tmin, Tmax, Tc, hh, ts, wsize, interval, occupation, xr=None):
    """
    permet l'habillage du graphique d'un épisode avec la zone de confort et les périodes d'occupation
    utilise la fonction Rectangle de matplotlib

    - Tmin : température minimale
    - Tmax : température maximale
    - Tc : température de consigne
    - hh : demi-intervalle de la zone de confort
    - nombre de points dans l'épisode
    - interval : pas de temps
    - occupation : agenda d'occupation avec 4 jours supplémentaires pour calculer les durées d'occupation

    retourne les objets matérialisant les zones de confort et d'occupation
    """
    if xr is None:
        xrs = np.arange(ts, ts+wsize*interval, interval)
        xr = np.array(xrs, dtype='datetime64[s]')
    zoneconfort = Rectangle((xr[0], Tc-hh), xr[-1]-xr[0], 2*hh, facecolor='g', alpha=0.5, edgecolor='None', label="zone de confort")

    changes = []
    for i in range(wsize):
        if occupation[i]==0 and occupation[i+1]!=0:
            changes.append(i)
    zonesOcc = []
    for i in changes:
        if i < wsize-1:
            l = getLevelDuration(occupation, i+1)
            Tmax = max(Tc+hh,Tmax)
            Tmin = min(Tc-hh,Tmin)
            h = Tmax - Tmin
            w = l * (xr[1]-xr[0])
            v = Rectangle((xr[i], Tmin), w, h, facecolor='orange', alpha=0.5, edgecolor='None')
            zonesOcc.append(v)

    return xr, zoneconfort, zonesOcc

class Vacancy(gym.Env):
    def __init__(self, Text, max_power, Tc, k, **model):
        """
        _Text : objet PyFina de température extérieure

        state : tableau numpy de 4 éléments :
        - Text
        - Tint
        - Tc * occupation
        - nb hours -> occupation change (from occupied to empty and vice versa)
        """
        super().__init__()
        self._Text = Text
        self._tss = Text.start
        self._tse = Text.start + Text.step * Text.shape[0]
        self._interval = Text.step
        self._max_power = max_power
        self._Tc = Tc
        self._k = k
        self._model = modelRC
        if model:
            self._model = model
        # la constante de temps du modèle électrique équivalent
        self._tcte = self._model["R"]*self._model["C"]
        self._cte = math.exp(-self._interval/self._tcte)
        self.action_space = spaces.Discrete(2)
        high = np.finfo(np.float32).max
        self.observation_space = spaces.Box(-high, high, (4,), dtype=np.float32)
        #print(self.observation_space)
        self.state = None
        self._label = "vacancy"

    def _reset(self, ts = None, Tint = None, seed: Optional[int] = None):
        """
        reset générique

        Avant d'appeler cette méthode, il faut fixer la taille de l'épisode _wsize !

        permet de définir :
        - self._xr : échelle de temps de l'épisode au format humain
        - self._pos : position de l'épisode dans la timesérie Text
        - self._tsvrai : timestamp du début de l'épisode

        initialise :
        - self.i : compteur de pas dans l'épisode
        - self._reward : récompense accumulée au cours de l'épisode
        - self._Tint : tableau numpy des températures intérieures au cours de l'épisode, de taille wsize + 1
        - self._Qc : tableau numpy de la puissance consommée au cours de l'épisode, de taille wsize + 1

        retourne self.state
        """
        self.seed(seed)
        if ts is None:
            start = self._tss
            tse = self._tse
            end = tse - self._wsize * self._interval - 4*24*3600
            # on tire un timestamp avant fin mai OU après début octobre
            ts = getRandomStart(start, end, 10, 5)
        self.i = 0
        self._pos = (ts - self._tss) // self._interval
        self._tsvrai = self._tss + self._pos * self._interval
        #print("episode timestamp : {}".format(self._tsvrai))
        # x axis = time for human
        xrs = np.arange(self._tsvrai, self._tsvrai + self._wsize * self._interval, self._interval)
        self._xr = np.array(xrs, dtype='datetime64[s]')
        Text = self._Text[self._pos]
        if not isinstance(Tint, (int, float)):
            Tint = random.randint(17,20)
        self._Tint = np.zeros(self._wsize + 1)
        self._Qc = np.zeros(self._wsize + 1)
        self._Tint[self.i] = Tint
        Tc, nbh = self.updateNonPhysParams()
        self.state = np.array([Text, Tint, Tc, nbh], dtype=np.float32)
        self._reward = 0
        return self.state

    def _render(self, zoneconfort = None, zonesOcc = None, stepbystep = True, label = None):
        """
        render générique
        """
        if self.i == 0 or not stepbystep:
            self._fig = plt.figure()
            self._ax1 = plt.subplot(311)
            self._ax2 = plt.subplot(312, sharex=self._ax1)
            self._ax3 = plt.subplot(313, sharex=self._ax1)
            if stepbystep :
                plt.ion()
        title = "{} - score : {}".format(self._tsvrai, round(self._reward,2))
        if label is not None :
            title = "{}\n{}".format(title, label)
        self._fig.suptitle(title)
        self._ax1.clear()
        self._ax2.clear()
        self._ax3.clear()
        self._ax1.plot(self._xr, self._Text[self._pos:self._pos+self._wsize])
        self._ax2.plot(self._xr[0:self.i], self._Tint[0:self.i])
        self._ax3.plot(self._xr[0:self.i], self._Qc[0:self.i])
        if self.i :
            if zoneconfort is not None :
                self._ax2.add_patch(zoneconfort)
            if zonesOcc is not None :
                for v in zonesOcc:
                    self._ax2.add_patch(v)
        plt.show()
        if stepbystep:
            plt.pause(0.0001)

    def _step(self, action):
        """
        fonction step générique
        """
        err_msg = "{} is not a valid action".format(action)
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."
        # reward at state
        reward = self.reward(action)
        self._reward += reward
        # Qc at state
        Qc = action * self._max_power
        self._Qc[self.i] = action
        # indoor temps at next state
        Text = self._Text[self._pos+self.i:self._pos+self.i+2]
        delta = self._cte * ( Qc / self._model["C"] + Text[0] / self._tcte ) + Qc / self._model["C"] + Text[1] / self._tcte
        x = self.state[1] * self._cte + self._interval * 0.5 * delta
        self.i += 1
        self._Tint[self.i] = x
        # non physical parameters at next state
        Tc, nbh = self.updateNonPhysParams()
        # on met à jour state avec les données de next state
        self.state = np.array([Text[1], x, Tc, nbh], dtype=np.float32)
        # return reward at state
        return reward

    def updateNonPhysParams(self):
        """
        estimates :
        - Tc*occupation
        - nbh -> occupation change (from occupied to empty and vice versa)

        *Note that 1h30 has to be coded as 1.5*

        In the vacancy case, Tc*occupation is always 0
        """
        self._nbsteps = self._wsize - 1 if self.i == 0 else self._nbsteps - 1
        return 0, self._nbsteps * self._interval / 3600

    def reward(self, action):
        self._rewardLabel = "Vote_final_reward_only"
        reward = 0
        Tc = self._Tc
        if self.state[3] == 0 :
            # l'occupation du bâtiment commence
            # pour converger vers la température cible
            if self.state[1] >= Tc :
                reward = -15 * (self.state[1] - Tc - 1) - 20
            if self.state[1] < Tc :
                reward = -15 * (Tc - 1 - self.state[1]) - 20
            # le bonus énergétique
            if self.state[1] <= Tc + 1 and self.state[1] >= Tc - 3:
                reward += self._tot_eko * self._k * self._interval / 3600
        # calcul de l'énergie économisée
        if not action :
            self._tot_eko += 1
        return reward

    def reset(self, ts=None, Tint=None, seed: Optional[int] = None, wsize = None):
        if not isinstance(wsize, int):
            self._wsize = 63 * 3600 // self._interval
        else :
            self._wsize = wsize
        self._tot_eko = 0
        if not isinstance(Tint, (int, float)):
            Tint = 20
        return self._reset(ts = ts, Tint = Tint, seed = seed)

    def step(self, action):
        """
        renvoie state, reward pour previous state, done, _
        """
        reward = self._step(action)
        done = None
        if self._nbsteps == -1:
            done = True
        return self.state, reward, done, {}

    def render(self, stepbystep = True, label=None):
        self._render(stepbystep = stepbystep, label = label)

    def close(self):
        #plt.savefig("test.png")
        plt.close()

class Building(Vacancy):
    def __init__(self, Text, agenda, wsize, max_power, Tc, k, **model):
        super().__init__(Text, max_power, Tc, k, **model)
        self._agenda = agenda
        self._wsize = wsize
        self._label = "week"

    def updateNonPhysParams(self):
        Tc = self._agenda[self._pos+self.i] * self._Tc
        occupation = self._agenda[self._pos+self.i:self._pos+self.i+self._wsize+4*24*3600//self._interval]
        nbh = getLevelDuration(occupation, 0) * self._interval / 3600
        return Tc, nbh

    def reward(self, action):
        # reward at (state, action)
        reward = 0
        Tc = self._Tc
        if self.state[2] != 0:
            l0 = Tc - 5
            l1 = Tc - 3
            l2 = Tc - 1
            if abs(self.state[1] - Tc) > 1:
                reward -= abs( self.state[1] - Tc) * self._interval / 3600
            if self._agenda[self._pos+self.i-1] == 0:
                if self.state[1] < l0:
                    reward -= 30
                if self.state[1] < l1:
                    reward -= 30
                if self.state[1] < l2:
                    reward -= 20
        else:
            if action :
                reward -= (self._k + max(0, self.state[1] - Tc)) * self._interval / 3600
            else :
                self._tot_eko += 1
        return reward

    def reset(self, ts=None, Tint=None, seed: Optional[int] = None):
        self._tot_eko = 0
        return self._reset(ts = ts, Tint = Tint, seed = seed)

    def step(self, action):
        reward = self._step(action)
        done = None
        if self.i == self._wsize - 1:
            done = True
        return self.state, reward, done, {}

    def render(self, stepbystep = True, label=None):
        if self.i:
            Tmin = np.min(self._Tint[0:self.i])
            Tmax = np.max(self._Tint[0:self.i])
            occupation = self._agenda[self._pos:self._pos+self._wsize+4*24*3600//self._interval]
            _, zoneconfort, zonesOcc = covering(Tmin, Tmax, self._Tc, 1, self._tsvrai, self._wsize, self._interval, occupation, self._xr)
            self._render(zoneconfort = zoneconfort, zonesOcc = zonesOcc, stepbystep = stepbystep)
        else:
            self._render(stepbystep = stepbystep)
