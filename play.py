#!/usr/bin/env python3
"""
joue des épisodes / produit des stats
"""

from EnergyGym import Environnement, Evaluate
import os
import matplotlib.pyplot as plt
import click

# le circuit
interval = 1800
# nombre d'intervalles sur lequel la simulation sera menée
wsize = 1 + 8*24*3600//interval
dir = "/var/opt/emoncms/phpfina"
import numpy as np
schedule = np.array([ [7,17], [7,17], [7,17], [7,17], [7,17], [-1,-1], [-1,-1] ])
Cw = 1162.5 #Wh/m3/K
max_power = 5 * Cw * 15

circuit = {"Text":1, "dir": dir,
           "schedule": schedule, "interval": interval, "wsize": wsize}

# on importe les configurations existantes de modèles depuis le fichier conf
from conf import models

hh = 1
optimalPolicies = ["intermittence", "occupation_permanente"]
# série d'épisodes dont on veut avoir des replays
# les 3 premiers : froid
# les 3 suivants : très froids
# le dernier : mi-saison
cold = [1577259140, 1605781940, 1608057140, 1610019140, 1612513940, 1611984740, 1633350740]

class EnvHyst(Environnement):
    def play(self, datas):
        """
        politique optimale de type hysteresys avec un modèle déterministe
        """
        # doit-on mettre en route le chauffage à l'étape 0 ?
        action = datas[0,2] <= self._Tc
        datas[0,0] = action * self._max_power

        # itération de l'étape 1 à la fin de l'épisode
        for i in range(1, datas.shape[0]):
            # l'état précédant est totalement déterminé, on peut calculer la température à l'état i
            datas[i,2] = self.sim(datas, i)
            # doit-on mettre en route le chauffage à l'étape i ?
            if datas[i,2] > self._Tc + self._hh or datas[i,2] < self._Tc - self._hh :
                action = datas[i,2] <= self._Tc
                datas[i,0] = action * self._max_power
            else:
                # on est dans la fenêtre > on ne change rien :-)
                datas[i,0] = datas[i-1,0]

        return datas

class EnvHystNocc(Environnement):
    def play(self, datas):
        """
        politique optimale en intermittence d'occupation avec un modèle déterministe

        les régulateurs industriels essayent de tendre vers cette politique optimale, mais les réglages sont complexes
        """
        # doit-on mettre en route le chauffage à l'étape 0 ?
        if datas[0,3] == 0:
            Tint_sim = self.sim2Target(datas,0)
            action = Tint_sim[-1] <= self._Tc
        else :
            action = datas[0,2] <= self._Tc
        datas[0,0] = action * self._max_power

        # itération de l'étape 1 à la fin de l'épisode
        for i in range(1,datas.shape[0]):
            #  calcul de la température à l'état i
            datas[i,2] = self.sim(datas, i)
            # doit-on mettre en route le chauffage à l'étape i ?
            if datas[i,3] == 0 :
                # pas d'occupation - calcul à la cible
                Tint_sim = self.sim2Target(datas, i)
                action = Tint_sim[-1] <= self._Tc
                datas[i,0] = action * self._max_power
            else:
                # en occupation
                # hystérésis classique
                if datas[i,2] > self._Tc + self._hh or datas[i,2] < self._Tc - self._hh :
                    action = datas[i,2] <= self._Tc
                    datas[i,0] = action * self._max_power
                else:
                    # on est dans la fenêtre > on ne change rien :-)
                    datas[i,0] = datas[i-1,0]

        return datas

class EvalHyst(Evaluate):
    def reward(self, datas, i):
        p = self._policy
        self._rewards[p]["confort"][i] = self._rewards[p]["confort"][i-1]
        Tc = self._env._Tc
        reward = - abs(datas[i,2] - Tc) * self._env._interval / 3600
        self._rewards[p]["confort"][i] += reward
        return reward

class EvalVote(Evaluate):
    def reward(self, datas, i):
        p = self._policy
        reward_architecture = ["confort", "vote", "energy", "gaspi"]
        reward = {}
        for r in reward_architecture:
            self._rewards[p][r][i] = self._rewards[p][r][i-1]
            reward[r] = 0
        Tc = self._env._Tc

        if datas[i,3] != 0:
            #Tc = datas[i,3]
            l0 = Tc - 5 * self._env._hh
            l1 = Tc - 3 * self._env._hh
            l2 = Tc - self._env._hh
            if abs(datas[i,2] - Tc) > self._env._hh:
                reward["confort"] = - abs( datas[i,2] - Tc) * self._env._interval / 3600
            if datas[i-1,3] == 0:
                if datas[i,2] < l0:
                    reward["vote"] -= 30
                if datas[i,2] < l1:
                    reward["vote"] -= 30
                if datas[i,2] < l2:
                    reward["vote"] -= 20
        else:
            if datas[i,0] :
                reward["energy"] = - self._k * self._env._interval / 3600
                # pénalite pas adaptée si le bâtiment est tellement déperditif et son système de chauffage si sous_dimensionné
                # qu'il lui faut parfois lorsqu'il fait très froid chauffer au dessus de la consigne hors occupation
                # pour pouvoir être sur d'avoir la consigne à l'ouverture
                reward["gaspi"] = - max(0, datas[i,2] - Tc) * self._env._interval / 3600

        for r in reward_architecture:
            self._rewards[p][r][i] += reward[r]

        result = sum(reward.values())
        return result

def load_agent(agent_path):
    import tensorflow as tf
    # custom_objects est nécessaire pour charger certains réseaux entrainés sur le cloud, via les github actions
    agent = tf.keras.models.load_model(agent_path, compile = False, custom_objects={'Functional':tf.keras.models.Model})
    return agent

def load(agent_path, ctxobj, silent = True):
    agent = load_agent(agent_path)
    tc = ctxobj['tc']
    model = ctxobj['model']
    optimalpolicy = ctxobj['optimalpolicy']
    max_power = ctxobj['max_power']

    from EnergyGym import getTruth
    Text, agenda = getTruth(circuit, visualCheck = not silent)
    print("max_power is {}".format(max_power))

    if optimalpolicy == "occupation_permanente":
        env = EnvHyst(Text, agenda, wsize, max_power, tc, hh, **model)
    elif optimalpolicy == "intermittence":
        env = EnvHystNocc(Text, agenda, wsize, max_power, tc, hh, **model)

    return agent, env

def snapshots(storage, agent_name, agent_path, ctxobj):
    """
    crée des snapshots pour chacun des épisodes présents dans la variable cold

    storage : répertoire dans lequel les snapshots seront enregistrés, du type agent_folder/snapshots
    """
    tc = ctxobj['tc']
    optimalpolicy = ctxobj['optimalpolicy']
    # modelkey est la clé permettant d'accéder à la configuration du modèle utilisé pour décrire l'environnement
    modelkey = ctxobj['modelkey']
    sub = "{}/{}".format(storage,agent_name.replace(".h5",""))
    if not os.path.isdir(sub):
        os.mkdir(sub)

    agent, env = load(agent_path, ctxobj)
    sandbox = Training(agent_path, env, agent, N=1)
    for ts in cold:
        sandbox.play(silent=True, ts=ts, snapshot=True)
        plt.savefig("{}/{}_{}_{}".format(sub,ts,modelkey,optimalpolicy))
        plt.close()

    # reconstruit le fichier markdown en inspectant le répertoire snapshot
    f = open("{}/README.md".format(sub), "w")
    f.write("# {}\n".format(agent_name.replace(".h5","")))
    all = os.listdir(sub)
    all.sort()
    family = {}
    for policy in optimalPolicies :
        family[policy] = []
    for a in all:
        if a.endswith(".png"):
            root = True
            for policy in optimalPolicies :
                if a.replace(".png","").endswith(policy) :
                    family[policy].append("![]({})".format(a))
                    root = False
            if root:
                f.write("![]({})\n".format(a))
    for policy in family:
        if family[policy] :
            f.write("\n# {}\n".format(policy))
            f.write("\n".join(family[policy]))
    f.close()

@click.group()
@click.option('--t_ext', type=int, default=1, prompt='numéro du flux de température extérieure ?')
@click.option('--model', type=click.Choice(models), prompt='modèle ?')
@click.option('--powerlimit', type=float, default=1, prompt='coefficient de réduction de la puissance max ?')
@click.option('--tc', type=int, default=20, prompt='température de consigne en °C')
@click.option('--n', type=int, prompt='nombre d\'épisodes à jouer, 0 = mode snapshot : le système joue des épisodes prédéfinis')
@click.option('--optimalpolicy', type=click.Choice(optimalPolicies), prompt='politique du modèle ?')
@click.option('--hystpath', type=str, default=None)
@click.pass_context
def main(ctx, t_ext, model, powerlimit, tc, n, optimalpolicy, hystpath):
    ctx.ensure_object(dict)
    ctx.obj['tc'] = tc
    ctx.obj['n'] = n
    ctx.obj['optimalpolicy'] = optimalpolicy
    ctx.obj['occupation_agent_path'] = None
    if hystpath is not None :
        from EnergyGym import pickName
        _, saved = pickName(name=hystpath)
        if saved :
            ctx.obj['occupation_agent_path'] = hystpath
    ctx.obj['modelkey'] = model
    ctx.obj['model'] = models[model]
    ctx.obj['max_power'] = powerlimit * max_power

    global circuit
    circuit["Text"] = t_ext

@main.command()
@click.option('--holiday', type=int, default=0, prompt='nombre de jours fériés à intégrer dans les replay')
@click.option('--silent', type=bool, prompt='silent mode = sans montrer les replays des épisodes ?')
@click.option('--k', type=float, default=0.9, prompt='paramètre énergie')
@click.pass_context
def play(ctx, holiday, silent, k):
    print(ctx.obj)
    tc = ctx.obj['tc']
    n = ctx.obj['n']
    optimalpolicy = ctx.obj['optimalpolicy']
    occupation_agent_path = ctx.obj['occupation_agent_path']
    print(holiday, silent, tc, n, optimalpolicy, k)
    input("press")
    if holiday:
        import random
        days = [0,4] if holiday == 1 else [0,1,2,3,4]
        holidays = []
        for i in range(holiday):
            tirage = random.choice(days)
            if tirage not in holidays:
                holidays.append(tirage)
        for i in holidays:
            circuit["schedule"][i] = [-1,-1]

    # demande à l'utilisateur un nom de réseau
    from EnergyGym import pickName
    agent_path, saved = pickName()

    if saved == True:
        if not n:
            agent_name = agent_path[agent_path.rfind("/")+1:]
            agent_folder = agent_path[:agent_path.rfind("/")]
            storage = "{}/snapshots".format(agent_folder)
            if not os.path.isdir(storage):
                os.mkdir(storage)
            snapshots(storage, agent_name, agent_path, ctx.obj)
        else:
            agent, env = load(agent_path, ctx.obj, silent=silent)
            if optimalpolicy == "occupation_permanente":
                sandbox = EvalHyst(agent_path, env, agent, N=n, k=k)
            elif optimalpolicy == "intermittence":
                sandbox = EvalVote(agent_path, env, agent, N=n, k=k)
                if occupation_agent_path is not None:
                    sandbox.setOccupancyAgent(load_agent(occupation_agent_path))

            sandbox.play(silent=False, ts=1641828600) # 10 janvier 2022

            #sandbox.play(silent=False, ts=1576559067) # 2019-12-17 06:04:27:+0100
            #sandbox.play(silent=False, ts=1577269940) # 2019-12-25 11:32:20:+0100
            #sandbox.play(silent=False, ts=1578983540) # 14 janvier 2020
            #sandbox.play(silent=False, ts=1579254245) # 2020-01-17 10:44:05:+0100
            #sandbox.play(silent=False, ts=1586850023) # 2020-04-14 09:40:23:+0200
            #sandbox.play(silent=False, ts=1589644200) # 2020-05-16 17:50:00:+0200

            #adatas = sandbox.play(silent=silent, ts=1603499540) # 2020-10-24 02:32:20:+0200
            #print(adatas)
            #sandbox.play(silent=False, ts=1606343540) # 2020-11-25 23:32:20:+0100
            #sandbox.play(silent=False, ts=1608928315) # 2020-12-25 21:31:55:+0100
            #sandbox.play(silent=False, ts=1610494340) # 2021-01-13 00:32:20:+0100
            #sandbox.play(silent=False, ts=1617370780) # 2021-04-02 15:39:40:+0200

            sandbox.run(silent=silent)
            sandbox.close(suffix=optimalpolicy)

if __name__ == "__main__":
    main()
