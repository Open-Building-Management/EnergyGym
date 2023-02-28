"""heatgym"""
import random
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import gym
from gym import spaces
import numpy as np

from .planning import get_random_start, get_level_duration

custom_gym_envs = [
    "Hyst",
    "Reduce",
    "Vacancy",
    "LSTMVacancy",
    "Building"
]

# --------------------------------------------------------------------------- #
# pdoc related
# --------------------------------------------------------------------------- #
_custom_gym_envs = [
    *custom_gym_envs,
    "Env",
    "StepRewardVacancy",
    "TopLimitVacancy"
]

vars_to_exclude_from_pdoc = [
    "action_space",
    "observation_space",
    "metadata",
    "render_mode",
    "spec"
]

__pdoc__ = {}
for cge in _custom_gym_envs:
    for excluded in vars_to_exclude_from_pdoc:
        __pdoc__[f'{cge}.{excluded}'] = False
# --------------------------------------------------------------------------- #

# modèle R1C1 par défault obtenu par EDW avec les données de Marc Bloch
#MODELRC = {"R": 2.54061406e-04, "C": 9.01650468e+08}
MODELRC = {"R": 5.94419964e-04, "C": 5.40132642e+07}
# pylint: disable=W0221

def confort(xr, tc, hh):
    """construit le rectangle vert de la zone de confort thermique

    xr : échelle de temps de l'épisode au format humain,
    tc : température de consigne,
    hh : demi-intervalle de la zone de confort
    """
    # facecolor='g': green color
    return Rectangle((xr[0], tc-hh), xr[-1]-xr[0], 2 * hh,
                     facecolor='g', alpha=0.5, edgecolor='None',
                     label="zone de confort")


def presence(xr, occupation, wsize, tmin, tmax, tc, hh):
    """construit les rectangles orange matérialisant l'occupation

    xr : échelle de temps de l'épisode au format humain,
    occupation : agenda d'occupation avec 4 jours supplémentaires
    pour calculer les nombres de pas jusqu'au prochain cgt d'occupation,
    wsize : nombre de points dans l'épisode,
    tmin : température minimale,
    tmax : température maximale,
    tc : température de consigne,
    hh : demi-intervalle de la zone de confort
    """
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


class Env(gym.Env):
    """base environnement"""
    def __init__(self, text, max_power, tc, **model):
        """
        text : objet PyFina de température extérieure

        max_power : puissance max de chauffage dispo en W

        tc : température de consigne

        model : dictionnaire du modèle d'environnement avec
        ses paramètres électriques R et C,
        action_space (taille de l'espace d'actions),
        nbh (nombre d'heures de l'historique),
        nbh_forecast (nombre d'heures de prévisions météo),
        k (coefficient énergétique),
        p_c (pondération sur le confort),
        vote_interval (intervalle de température autour de la consigne
        dans lequel la récompense énergétique est attribuée)

        la taille de l'espace d'observation dépend de nbh et nbh_forecast
        et doit être fixée dans les classes filles
        """
        super().__init__()
        self.text = text
        self._tss = text.start
        self._tse = text.start + text.step * text.shape[0]
        # pas de temps en secondes
        self._interval = text.step
        # nombre de pas dans un épisode (taille de la fenêtre)
        self.wsize = None
        # espace d'actions
        self.action_space = spaces.Discrete(model.get("action_space", 2))
        # nbh est le nombre d'heures qu'on peut remonter dans le passé
        self.nbh = int(model.get("nbh", 0))
        # nombre de points de l'histoire passée y compris le point courant
        self.pastsize = int(1 + self.nbh * 3600 // self._interval)
        # nbh_forecast est le nombre d'heures qu'on peut prévoir dans le futur
        self.nbh_forecast = int(model.get("nbh_forecast", 0))
        # tableaux numpy des températures passées
        # et de l'historique de consommation
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
        self.tot_reward = None
        # tableau numpy des températures intérieures au cours de l'épisode
        # de taille wsize + 1
        # dimensionné pour que tint[-1] soit la température à l'ouverture
        # entre 0 et wsize, il y a wsize intervalles et wsize + 1 éléments
        self.tint = None
        # tableau numpy des actions énergétiques au cours de l'épisode
        # de taille wsize + 1
        self.action = None
        # nombre de pas de temps sans conso énergétique pour l'épisode
        self.tot_eko = 0
        self.max_power = max_power
        self.tc = tc
        # tc_episode permet d'entrainer à consigne variable
        self.tc_episode = None
        self.reduce = None
        # p_c : pondération du confort
        # k : coefficient énergie
        self._p_c = model.get("p_c", 15)
        self._vote_interval = model.get("vote_interval", (-3, 1))
        self._k = model.get("k", 0.9)
        self.model = model if model else MODELRC
        # calcule les constantes du modèle électrique équivalent
        self.tcte, self.cte = self._update_cte_tcte()
        # current state in the observation space
        self.state = None
        # paramètres pour le rendu graphique
        self._fig = None
        self._ax1 = None
        self._ax2 = None
        self._ax3 = None
        # agenda d'occupation
        self.agenda = None

    def _get_future(self):
        """construit le futur horaire
        en utilisant directement text"""
        hti = 3600 // self._interval
        indexes = np.arange(0, self.nbh_forecast * hti, hti)
        pos = self.pos + self.i + hti
        return self.text[pos + indexes]

    def _get_past(self):
        """construit le passé horaire
        en utilisant tint_past, text_past et q_c_past"""
        hti = 3600 // self._interval
        indexes = np.arange(0, 1 + self.nbh * hti, hti)
        q_c_past_horaire = np.zeros(self.nbh)
        for i in range(self.nbh):
            pos = i * hti
            val = np.mean(self.q_c_past[pos:pos+hti])
            q_c_past_horaire[i] = val
        return self.text_past[indexes], self.tint_past[indexes], q_c_past_horaire

    def _reset(self, ts=None, tint=None, tc_episode=None, tc_step=None):
        """
        generic reset method - private

        la méthode publique reset fixe la taille de l'épisode wsize,
        puis exécute _reset

        permet de définir self._xr, self.pos, self.tsvrai
        ces grandeurs sont des constantes de l'épisode

        initialise self.i, self.tot_reward, self.tint, self.action
        ces grandeurs sont mises à jour à chaque pas de temps

        retourne state
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
        # on fixe la température de consigne de notre épisode,
        # c'est-à-dire la température qu'il doit faire quant le bâtiment est occupé
        # si on veut fonctionner/entrainer à consigne variable,
        # on tire au sort une valeur et on la passe en argument à reset()
        self.tc_episode = self.tc
        if isinstance(tc_episode, (int, float)):
            self.tc_episode = tc_episode
        # x axis = time for human
        xrs = np.arange(self.tsvrai,
                        self.tsvrai + (self.wsize+1) * self._interval,
                        self._interval)
        self._xr = np.array(xrs, dtype='datetime64[s]')
        self.tint = np.zeros(self.wsize + 1)
        self.action = np.zeros(self.wsize + 1)
        # construction d'une histoire passée
        if not isinstance(tint, (int, float)):
            tint = self.tc_episode + random.randint(-3, 0)
        self.tint_past[0] = tint
        action = self.tint_past[0] <= self.tc_episode
        q_c = action * self.max_power
        self.text_past = self.text[self.pos + 1 - self.pastsize:self.pos + 1]
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
        self.state = self._state(tc=tc_step)
        self.tot_reward = 0
        return self.state

    def _render(self, zone_confort=None, zones_occ=None,
                stepbystep=True,
                label=None, extra_datas=None,
                snapshot=False):
        """generic render method"""
        if self.i == 0 or not stepbystep:
            self._fig = plt.figure(figsize=(20,10))
            self._ax1 = plt.subplot(311)
            self._ax2 = plt.subplot(312, sharex=self._ax1)
            self._ax3 = plt.subplot(313, sharex=self._ax1)
            if stepbystep :
                plt.ion()
        title = f'{self.tsvrai} - score: {self.tot_reward:.2f} - tc_episode: {self.tc_episode}°C'
        if label is not None :
            title = f'{title}\n{label}'
        self._fig.suptitle(title)
        self._ax1.clear()
        self._ax2.clear()
        self._ax3.clear()
        self._ax1.plot(self._xr, self.text[self.pos:self.pos+self.wsize+1])
        self._ax2.plot(self._xr[0:self.i], self.tint[0:self.i])
        self._ax3.plot(self._xr[0:self.i], self.action[0:self.i])
        # données externes
        if extra_datas is not None and not stepbystep:
            self._ax2.plot(self._xr[0:self.wsize], extra_datas[:,1], color="orange")
            energy = extra_datas[:,0] * (self.action_space.n - 1)
            self._ax3.plot(self._xr[0:self.wsize], energy, color="orange")
        if self.i :
            if zone_confort is not None :
                self._ax2.add_patch(zone_confort)
            if zones_occ is not None :
                for occ in zones_occ:
                    self._ax2.add_patch(occ)
        if not snapshot:
            plt.show()
        if stepbystep:
            plt.pause(0.0001)

    def _step(self, action, tc_step=None):
        """generic step method"""
        err_msg = f'{action} is not a valid action'
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."
        # reward at state
        reward = self.reward(action)
        self.tot_reward += reward
        # q_c at state
        q_c = action * self.max_power / (self.action_space.n - 1)
        self.action[self.i] = action
        # indoor temp at next state
        text = self.text[self.pos+self.i:self.pos+self.i+2]
        delta = self.cte * (q_c / self.model["C"] + text[0] / self.tcte)
        delta += q_c / self.model["C"] + text[1] / self.tcte
        tint = self.tint[self.i] * self.cte + self._interval * 0.5 * delta
        self.i += 1
        done = False
        if self.i <= self.wsize:
            self.tint[self.i] = tint
            # on met à jour state avec les données de next state
            self.tint_past = np.array([*self.tint_past[1:], tint])
            self.text_past = np.array([*self.text_past[1:], text[1]])
            if self.pastsize > 1:
                self.q_c_past = np.array([*self.q_c_past[1:], q_c])
            self.state = self._state(tc=tc_step)
        else:
            self.state = None
            done = True
        # return reward, done at state
        return reward, done

    def _state(self, tc=None):
        """return the current state after all calculations are done"""
        if tc is None:
            tc=self.tc_episode
        text_future_horaire = self._get_future()
        text_past_horaire, tint_past_horaire, q_c_past_horaire = self._get_past()
        return np.array([*text_past_horaire,
                         *text_future_horaire,
                         *tint_past_horaire,
                         *q_c_past_horaire/self.max_power,
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
            zones_occ = presence(self._xr, occupation, self.wsize,
                                 tmin, tmax, self.tc_episode, 1)
        return zone_confort, zones_occ

    def _update_cte_tcte(self):
        """met à jour les constantes du modèle"""
        tcte = self.model["R"] * self.model["C"]
        cte = math.exp(-self._interval/tcte)
        return tcte, cte

    def _eko(self, action):
        """économie d'énergie associée à une action
        exprimée en pourcentage d'un pas de temps mobilisant la puissance maxi
        """
        return round(1 - action / (self.action_space.n - 1), 1)

    def set_agenda(self, agenda):
        """définit l'agenda d'occupation"""
        self.agenda = agenda

    def set_reduce(self, reduce):
        """fixe le nombre de degrés en moins sur la consigne hors occupation"""
        self.reduce = reduce

    def update_model(self, model):
        """met à jour les paramètres R et C du modèle
        et les constantes associées"""
        original = self.model
        for param in ["R", "C"]:
            self.model[param] = model.get(param, original[param])
        self.tcte, self.cte = self._update_cte_tcte()

    def reward(self, action):
        """récompense hystéresis
        ACCORDEE A CHAQUE PAS DE TEMPS"""
        reward = 0
        tc = self.tc_episode
        tint = self.tint[self.i]
        # on pondère car on peut entrainer à des pas de temps différents
        # cette pondération n'a d'impact si on entraine tjrs à l'heure
        # cette pondération n'a pas d'influence sur la convergence à priori
        # MAIS pour une performance donnée,
        # celà permet de rester sur des niveaux de récompenses équivalents,
        # même si on décide de changer le pas de temps lors des entrainements
        reward = - abs(tint - tc) * self._interval / 3600
        # calcul de l'énergie économisée
        self.tot_eko += self._eko(action)
        return reward

    def reset(self, ts=None,
              tint=None, tc_episode=None, tc_step=None,
              wsize=None):
        """episode reset"""
        if not isinstance(wsize, int):
            self.wsize = 63 * 3600 // self._interval
        else :
            self.wsize = int(wsize)
        self.tot_eko = 0
        return self._reset(ts=ts, tint=tint, tc_episode=tc_episode, tc_step=tc_step)

    def step(self, action, tc_step=None):
        """retourne un tuple :

        - state,

        - la récompense pour le previous state,

        - un booléen done (true si l'épisode est fini, false sinon)

        - un dictionnaire vide qui peut contenir des observations
        """
        reward, done = self._step(action, tc_step=tc_step)
        return self.state, reward, done, {}

    def render(self, stepbystep=True,
               label=None, extra_datas=None,
               snapshot=False):
        """render realtime or not

        on ne fait pas appel à _covering car en mode Vacancy, on n'a besoin
        d'aucun zonage du graphique
        """
        self._render(stepbystep=stepbystep,
                     label=label, extra_datas=extra_datas,
                     snapshot=snapshot)

    def close(self):
        """closing"""
        #plt.savefig("test.png")
        plt.close()

# --------------------------------------------------------------------------- #
# ready to use implementations
# il faut définir l' espace d'observation
# --------------------------------------------------------------------------- #
class Hyst(Env):
    """mode hystéresis permanent

    state est un vecteur de taille 3 * nbh + 2 + nbh_forecast + 1

    pour le construire, on met bout à bout
    l'historique de température extérieure de taille nbh+1,
    les prévisions de température extérieure de taille nbh_forecast,
    l'historique de température intérieure de taille nbh+1,
    l'historique de chauffage de taille nbh,
    la consigne de température intérieure

    si nbh=0 et nbh_forecast=0, l'espace d'observation est de taille 3
    [Text, Tint, tc]
    """
    def __init__(self, text, max_power, tc, **model):
        """ready to use gym environment"""
        super().__init__(text, max_power, tc, **model)
        high = np.finfo(np.float32).max
        self.observation_space = spaces.Box(-high, high,
                                            (3*self.nbh+2+self.nbh_forecast+1,),
                                            dtype=np.float32)

    def render(self, stepbystep=True,
               label=None, extra_datas=None,
               snapshot=False):
        """avec affichage des zones de confort et d'occupation"""
        if self.i:
            zone_confort, zones_occ = self._covering()
            self._render(zone_confort=zone_confort, zones_occ=zones_occ,
                         stepbystep=stepbystep,
                         label=label, extra_datas=extra_datas,
                         snapshot=snapshot)
        else:
            self._render(stepbystep=stepbystep, label=label)


class Reduce(Hyst):
    """mode hystérésis avec réduit hors occupation"""
    def _state(self, tc=None):
        if tc is None:
            tc = self.tc_episode
        if self.agenda[self.pos + self.i] == 0:
            tc -= self.reduce
        return super()._state(tc=tc)


class Vacancy(Env):
    """mode hors occupation

    state est un vecteur de taille 3 * nbh + 2 + nbh_forecast + 2

    par rapport au cas hystérésis, on rajoute le nombre d'heures
    d'içi le prochain changement d'occupation
    """
    def __init__(self, text, max_power, tc, **model):
        """ready to use gym environment"""
        super().__init__(text, max_power, tc, **model)
        high = np.finfo(np.float32).max
        self.observation_space = spaces.Box(-high, high,
                                            (3*self.nbh+2+self.nbh_forecast+2,),
                                            dtype=np.float32)
        #print(self.observation_space)

    def _state(self, tc=None):
        if tc is None:
            tc = self.tc_episode
        # nbh -> occupation change (from occupied to empty and vice versa)
        # Note that 1h30 has to be coded as 1.5
        nbh = (self.wsize - self.i) * self._interval / 3600
        result = super()._state(tc=tc)
        return np.array([*result, nbh], dtype=np.float32)

    def reward(self, action):
        """JUST A final reward mixing confort and energy"""
        reward = 0
        tc = self.tc_episode
        tint = self.tint[self.i]
        if self.i == self.wsize:
            # l'occupation du bâtiment commence
            # pour converger vers la température cible
            #print(tint, tc, self.tot_eko, self._interval)
            # on ne pondère pas la récompense en température
            # car elle est purement ponctuelle, acquise uniquement à l'ouverture
            reward = - self._p_c * abs(tint - tc)
            # le bonus énergétique
            # on pondère car même s'il s'agit d'une récompense finale,
            # elle est acquise sur toute la durée de l'épisode
            vmin = self._vote_interval[0]
            vmax = self._vote_interval[1]
            if vmin <= tint - tc <= vmax:
                reward += self._k * self.tot_eko * self._interval / 3600
        else:
            self.tot_eko += self._eko(action)
        return reward


class LSTMVacancy(Vacancy):
    """mode hors occupation

    state est une matrice 2D de taille (nbh, 5)

    en mode batch on aura donc affaire à une matrice 3D
    [batch, time, features]
    """
    def __init__(self, text, max_power, tc, **model):
        super().__init__(text, max_power, tc, **model)
        high = np.finfo(np.float32).max
        self.observation_space = spaces.Box(-high, high,
                                            (self.nbh, 5),
                                            dtype=np.float32)

    def _state(self, tc=None):
        if tc is None:
            tc = self.tc_episode
        text_past_horaire, tint_past_horaire, q_c_past_horaire = self._get_past()
        # nbh -> occupation change (from occupied to empty and vice versa)
        # Note that 1h30 has to be coded as 1.5
        nbh = (self.wsize - self.i) * self._interval / 3600
        nbh_past = np.arange(nbh + self.nbh - 1, nbh - 1, -1)
        result = np.array([
            text_past_horaire[1:],
            tint_past_horaire[1:],
            q_c_past_horaire,
            tc * np.ones(self.nbh),
            nbh_past
        ], dtype=np.float32)
        return result.transpose()


class StepRewardVacancy(Vacancy):
    """do not overheat trial 1"""
    def reward(self, action):
        """récompense à chaque step et non plus seulement finale"""
        reward = super().reward(action)
        if self.i != self.wsize:
            reward -= self._eko(action) * self._interval / 3600
        return reward


class TopLimitVacancy(Vacancy):
    """do not overheat trial 2"""
    def reward(self, action):
        reward = super().reward(action)
        if self.tint[self.i] > self.tc_episode + 1:
            reward -= (self.tint[self.i] - self.tc_episode - 1) * self._interval / 3600
        return reward


class Building(Vacancy):
    """alternance d'occupation et de non-occupation

    no real use for trainings

    pour jouer des semaines, lorsqu'on produit des stats

    à titre de comparaison, permet aussi d'utiliser un ancien agent
    2021_09_23_07_42_32_hys20_retrained_k0dot9_hys20.h5
    """
    def _state(self, tc=None):
        pos1 = self.pos + self.i
        if tc is None:
            tc = self.agenda[pos1] * self.tc_episode
        pos2 = self.pos + self.i + self.wsize + 4 * 24 * 3600 // self._interval
        occupation = self.agenda[pos1:pos2]
        nbh = get_level_duration(occupation, 0) * self._interval / 3600
        result = super()._state(tc=tc)
        return np.array([*result[:-1], nbh], dtype=np.float32)

    def reward(self, action):
        reward = 0
        if self.agenda[self.pos + self.i] != 0:
            tc = self.tc_episode
            tint = self.tint[self.i]
            reward -= abs(tint - tc) * self._interval / 3600
        self.tot_eko += self._eko(action)
        return reward

    def reset(self, ts=None, tint=None, tc_episode=None, tc_step=None, wsize=None):
        """building reset"""
        if not isinstance(wsize, int):
            self.wsize = 8*24*3600 // self._interval
        else :
            self.wsize = int(wsize)
        self.tot_eko = 0
        return self._reset(ts=ts, tint=tint, tc_episode=tc_episode, tc_step=tc_step)

    def render(self, stepbystep=True,
               label=None, extra_datas=None,
               snapshot=False):
        if self.i:
            zone_confort, zones_occ = self._covering()
            self._render(zone_confort=zone_confort, zones_occ=zones_occ,
                         stepbystep=stepbystep,
                         label=label, extra_datas=extra_datas,
                         snapshot=snapshot)
        else:
            self._render(stepbystep=stepbystep)
