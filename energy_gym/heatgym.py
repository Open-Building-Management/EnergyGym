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
    "StepRewardVacancy",
    "Building"
]

# --------------------------------------------------------------------------- #
# pdoc related
# --------------------------------------------------------------------------- #
_custom_gym_envs = [
    *custom_gym_envs,
    "Env",
    "D2Vacancy",
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

private = [
    "_get_future", "_get_past", "_reset", "_render", "_step",
    "_state", "_covering", "_update_cte_tcte", "_eko"
]

for method in private:
    __pdoc__[f'Env.{method}'] = True
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
        k (coefficient énergétique sur la partie finale de la récompense),
        k_step (coefficient énergétique sur la partie pas à pas de la récompense),
        p_c (pondération sur le confort),
        vote_interval (intervalle de température autour de la consigne
        dans lequel la récompense énergétique est attribuée),
        autosize_max_power (booléen à activer pour dimensionner
        la puissance maximale disponible en fonction de l'isolation),
        mean_prev (booléen à activer pour intégrer dans l'espace d'actions
        la temperature extérieure moyenne d'ici le prochain changement
        d'occupation, uniquement pour les scénarios de Type Vacancy)

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
        # économie d'énergie pour le maintien de tc_episode pendant tout l'épisode
        self.min_eko = 0
        self.limit = 0
        self.autosize_max_power = model.get("autosize_max_power", False)
        self.max_power = max_power
        self.tc = tc
        # tc_episode permet d'entrainer à consigne variable
        self.tc_episode = None
        self.reduce = None
        # p_c : pondération du confort
        # k : coefficient énergie
        self._p_c = model.get("p_c", 15)
        self._vote_interval = model.get("vote_interval", (-1, 1))
        self._k = model.get("k", 1)
        self._k_step = model.get("k_step", 1)
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
        self.mean_prev = model.get("mean_prev", False)
        self.mean_text_episode = None

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
        if self.autosize_max_power:
            # on dimensionne une puissance max théorique
            # pour maintenir 20°C à l'intérieur quant il fait 0°C dehors
            power = max(1, 20 * 1e-4 / self.model["R"])
            # on arrondit à la dizaine de KW
            # on applique un coefficient de sécurité de 10%
            self.max_power = 1.1 * round(power, 0) * 1e+4
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
        # solution optimale
        agenda = np.zeros(self.wsize + 1)
        agenda[self.wsize] = 1
        optimal_solution = play_hystnvacancy(self, self.pos, self.wsize,
                                             self.tint[0], self.tc_episode, 1,
                                             agenda=agenda)
        self.limit = self.wsize - np.sum(optimal_solution[:, 0])
        pos1 = self.pos
        pos2 = self.pos + self.wsize + 1
        self.mean_text_episode = np.mean(self.text[pos1:pos2])
        return self.state

    def _render(self, zone_confort=None, zones_occ=None,
                stepbystep=True,
                label=None, extra_datas=None,
                snapshot=False):
        """generic render method"""
        # si stepbystep est True, on ne crée la figure que si i vaut 0
        # si stepbystep est False, on la crée quel que soit i
        # en effet, dans ce cas, on appelle render à la fin de l'épisode, avec i > 0
        if self.i == 0 or not stepbystep:
            self._fig = plt.figure(figsize=(20,10))
            self._ax1 = plt.subplot(311)
            self._ax2 = plt.subplot(312, sharex=self._ax1)
            self._ax3 = plt.subplot(313, sharex=self._ax1)
            if stepbystep :
                plt.ion()
        title = f'{self.tsvrai}'
        title = f'{title} - text_moy_episode:{self.mean_text_episode:.2f}°C'
        title = f'{title} - tc_episode: {self.tc_episode}°C'
        title = f'{title} - score: {self.tot_reward:.2f}'
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
        # si stepbystep est True, on affiche l'image et le mode snapshot est sans effet
        # si snapshot est False, qu'on soit en mode stepbystep ou pas, on affiche l'image
        if stepbystep or not snapshot:
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
        """fixe le nombre de degrés en moins sur la consigne hors occupation,
        la valeur ainsi fixée, notée reduce, sert uniquement dans la classe fille Reduce
        """
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
        """reset avec des épisodes de 63 heures par défaut"""
        if not isinstance(wsize, int):
            self.wsize = 63 * 3600 // self._interval
        else :
            self.wsize = int(wsize)
        self.tot_eko = 0
        self.min_eko = 0
        return self._reset(ts=ts, tint=tint, tc_episode=tc_episode, tc_step=tc_step)

    def step(self, action, tc_step=None):
        """Avance d'un pas dans l'environnement en réalisant
        l'action fournie en paramètre

        retourne un tuple :

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
        """render realtime or not,
        avec ou sans affichage des zones de confort et d'occupation
        """
        if self.__class__.__name__ in ["Hyst", "Building", "Reduce"]:
            # affichage des zones de confort et d'occupation
            if self.i:
                zone_confort, zones_occ = self._covering()
                self._render(zone_confort=zone_confort, zones_occ=zones_occ,
                             stepbystep=stepbystep,
                             label=label, extra_datas=extra_datas,
                             snapshot=snapshot)
            else:
                self._render(stepbystep=stepbystep, label=label)
        else:
            # sans affichage des zones de confort et d'occupation
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
        """**ready to use gym environment**"""
        super().__init__(text, max_power, tc, **model)
        high = np.finfo(np.float32).max
        self.observation_space = spaces.Box(-high, high,
                                            (3*self.nbh+2+self.nbh_forecast+1,),
                                            dtype=np.float32)


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
        """**ready to use gym environment**"""
        super().__init__(text, max_power, tc, **model)
        high = np.finfo(np.float32).max
        if self.mean_prev:
            self.nbh_forecast = 0
        nb_observations = 3*self.nbh+2+self.nbh_forecast+2+self.mean_prev
        self.observation_space = spaces.Box(-high, high,
                                            (nb_observations,),
                                            dtype=np.float32)

    def _state(self, tc=None):
        if tc is None:
            tc = self.tc_episode
        # nbh -> occupation change (from occupied to empty and vice versa)
        # Note that 1h30 has to be coded as 1.5
        nbh = (self.wsize - self.i) * self._interval / 3600
        result = super()._state(tc=tc)[:-1]
        if self.mean_prev:
            pos1 = self.pos + self.i
            pos2 = self.pos + self.wsize + 1
            mean_text_to_target = np.mean(self.text[pos1:pos2])
            result = np.array([*result, mean_text_to_target], dtype=np.float32)
        result = np.array([*result, tc, nbh], dtype=np.float32)
        return result

    def reward(self, action):
        """JUST A final reward"""
        reward = 0
        tc = self.tc_episode
        tint = self.tint[self.i]
        if self.i == self.wsize:
            # l'occupation du bâtiment commence
            reward = - self._p_c * abs(tint - tc)

            vmin = self._vote_interval[0]
            vmax = self._vote_interval[1]
            peko = round(100 * self.tot_eko / self.wsize, 1)
            pmineko = round(100 * self.min_eko / self.wsize, 1)
            popteko = round(100 * self.limit / self.wsize, 1)
            #base_max = max(pmineko, popteko)
            #base_min = min(pmineko, popteko)
            # on arrondit à l'entier supérieur
            # pour tenir compte de l'imprécision du monitoring
            tint = round(tint)
            #if tint > tc + vmax and peko >= pmineko:
            #    reward = self._k * peko
            if vmin <= tint - tc <= vmax:
                #reward += self._k * self.tot_eko * self._interval / 3600
                #reward = self._k * peko
                # vu qu'on est dans la zone de confort
                # on ramène reward à zéro
                # pour à minima annuler la pénalité hystérésis
                reward = 0
                # si on a mieux bossé que la baseline
                # on rajoute un bonus énergétique
                if peko >= pmineko:
                    reward = self._k * (peko - pmineko)
        else:
            self.tot_eko += self._eko(action)
            text = self.text[self.pos + self.i]
            self.min_eko += 1
            if text < tc:
                self.min_eko -= (tc - text) / ( self.max_power * self.model["R"])
        return float(reward)


class D2Vacancy(Vacancy):
    """mode hors occupation

    state est une matrice 2D de taille (nbh, 5)

    les batchs seront donc des matrices 3D
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
    """do not overheat"""
    def reward(self, action):
        """récompense à chaque step et non plus seulement finale"""
        reward = super().reward(action)
        coeff = (self.wsize - self.i) * self._interval / self.tcte
        reward -= (1 - self._eko(action)) * coeff * self._k_step
        return reward


class TopLimitVacancy(Vacancy):
    """do not overheat trial 2"""
    def reward(self, action):
        reward = super().reward(action)
        if self.i < self.wsize:
            reward = -1 if action else 1
            if self.i >= self.limit :
                reward = 0
        return float(reward)


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
            vmin = self._vote_interval[0]
            vmax = self._vote_interval[1]
            if tint - tc < vmin or tint - tc > vmax :
                reward -= abs(tint - tc) * self._interval / 3600
        self.tot_eko += self._eko(action)
        return reward

    def reset(self, ts=None, tint=None, tc_episode=None, tc_step=None, wsize=None):
        """reset avec des épisodes de 8 jours par défaut"""
        if not isinstance(wsize, int):
            self.wsize = 8*24*3600 // self._interval
        else :
            self.wsize = int(wsize)
        self.tot_eko = 0
        return self._reset(ts=ts, tint=tint, tc_episode=tc_episode, tc_step=tc_step)
