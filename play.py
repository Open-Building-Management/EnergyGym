#!/usr/bin/env python3
"""joue des épisodes et produit des stats"""
import os
import random
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import click
from energy_gym import Environnement, Evaluate, get_truth, pick_name
# on importe les configurations existantes de modèles depuis le fichier conf
from conf import MODELS

# le circuit
INTERVAL = 1800
# nombre d'intervalles sur lequel la simulation sera menée
WSIZE = 1 + 8*24*3600//INTERVAL
PATH = "datas"

SCHEDULE = np.array([[7,17], [7,17], [7,17], [7,17], [7,17], [-1,-1], [-1,-1]])
CW = 1162.5 #Wh/m3/K
MAX_POWER = 5 * CW * 15

circuit = {"Text":1, "dir": PATH,
           "schedule": SCHEDULE, "interval": INTERVAL, "wsize": WSIZE}

hh = 1
OPTIMAL_POLICIES = ["intermittence", "occupation_permanente"]
# série d'épisodes dont on veut avoir des replays
# les 3 premiers : froid
# les 3 suivants : très froids
# le dernier : mi-saison
COLD = [1577259140, 1605781940, 1608057140, 1610019140, 1612513940, 1611984740, 1633350740]


class EnvHyst(Environnement):
    """politique optimale de type hystérésis avec un modèle déterministe"""
    def play(self, datas):
        # doit-on mettre en route le chauffage à l'étape 0 ?
        action = datas[0,2] <= self.tc
        datas[0,0] = action * self.max_power

        # itération de l'étape 1 à la fin de l'épisode
        for i in range(1, datas.shape[0]):
            # l'état précédant est totalement déterminé, on peut calculer la température à l'état i
            datas[i,2] = self.sim(datas, i)
            # doit-on mettre en route le chauffage à l'étape i ?
            if datas[i,2] > self.tc + self.hh or datas[i,2] < self.tc - self.hh :
                action = datas[i,2] <= self.tc
                datas[i,0] = action * self.max_power
            else:
                # on est dans la fenêtre > on ne change rien :-)
                datas[i,0] = datas[i-1,0]

        return datas


class EnvHystNocc(Environnement):
    """politique optimale en intermittence d'occupation avec un modèle déterministe"""
    def play(self, datas):
        # doit-on mettre en route le chauffage à l'étape 0 ?
        if datas[0,3] == 0:
            tint_sim = self.sim2target(datas,0)
            action = tint_sim[-1] <= self.tc
        else :
            action = datas[0,2] <= self.tc
        datas[0,0] = action * self.max_power

        # itération de l'étape 1 à la fin de l'épisode
        for i in range(1,datas.shape[0]):
            #  calcul de la température à l'état i
            datas[i,2] = self.sim(datas, i)
            # doit-on mettre en route le chauffage à l'étape i ?
            if datas[i,3] == 0 :
                # pas d'occupation - calcul à la cible
                tint_sim = self.sim2target(datas, i)
                action = tint_sim[-1] <= self.tc
                datas[i,0] = action * self.max_power
            else:
                # en occupation
                # hystérésis classique
                if datas[i,2] > self.tc + self.hh or datas[i,2] < self.tc - self.hh :
                    action = datas[i,2] <= self.tc
                    datas[i,0] = action * self.max_power
                else:
                    # on est dans la fenêtre > on ne change rien :-)
                    datas[i,0] = datas[i-1,0]

        return datas


class EvalHyst(Evaluate):
    """récompense hystérésis"""
    def reward(self, datas, i):
        policy = self._policy
        self._rewards[policy]["confort"][i] = self._rewards[policy]["confort"][i-1]
        tc = self._env.tc
        reward = - abs(datas[i,2] - tc) * self._env.interval / 3600
        self._rewards[policy]["confort"][i] += reward
        return reward


class EvalVote(Evaluate):
    """récompense pour une occupation intermittente"""
    def reward(self, datas, i):
        policy = self._policy
        reward_types = ["confort", "vote", "energy", "gaspi"]
        reward = {}
        for rewtyp in reward_types:
            self._rewards[policy][rewtyp][i] = self._rewards[policy][rewtyp][i-1]
            reward[rewtyp] = 0
        tc = self._env.tc

        if datas[i, 3] != 0:
            #tc = datas[i,3]
            l_0 = tc - 5 * self._env.hh
            l_1 = tc - 3 * self._env.hh
            l_2 = tc - self._env.hh
            if abs(datas[i, 2] - tc) > self._env.hh:
                reward["confort"] = - abs(datas[i, 2] - tc) * self._env.interval / 3600
            if datas[i-1, 3] == 0:
                if datas[i, 2] < l_0:
                    reward["vote"] -= 30
                if datas[i, 2] < l_1:
                    reward["vote"] -= 30
                if datas[i, 2] < l_2:
                    reward["vote"] -= 20
        else:
            if datas[i, 0] :
                reward["energy"] = - self._k * self._env.interval / 3600
                # pénalite pas adaptée si le bâtiment est tellement déperditif et son système de chauffage si sous_dimensionné
                # qu'il lui faut parfois lorsqu'il fait très froid chauffer au dessus de la consigne hors occupation
                # pour pouvoir être sur d'avoir la consigne à l'ouverture
                reward["gaspi"] = - max(0, datas[i,2] - tc) * self._env.interval / 3600

        for rewtyp in reward_types:
            self._rewards[policy][rewtyp][i] += reward[rewtyp]

        result = sum(reward.values())
        return result

def load_agent(agent_path):
    """load tensorflow network"""
    import tensorflow as tf
    # custom_objects est nécessaire pour charger certains réseaux entrainés sur le cloud, via les github actions
    agent = tf.keras.models.load_model(agent_path, compile = False, custom_objects={'Functional':tf.keras.models.Model})
    return agent

def load(agent_path, ctxobj, silent = True):
    """load tensorflow network and creates environment"""
    agent = load_agent(agent_path)
    tc = ctxobj['tc']
    model = ctxobj['model']
    optimalpolicy = ctxobj['optimalpolicy']
    max_power = ctxobj['max_power']

    text, agenda = get_truth(circuit, visual_check = not silent)
    print(f'max_power is {max_power}')

    if optimalpolicy == "occupation_permanente":
        env = EnvHyst(text, agenda, WSIZE, max_power, tc, hh, **model)
    elif optimalpolicy == "intermittence":
        env = EnvHystNocc(text, agenda, WSIZE, max_power, tc, hh, **model)

    return agent, env

def snapshots(storage, agent_name, ctxobj, sandbox):
    """
    crée des snapshots pour chacun des épisodes présents dans la variable COLD

    storage : répertoire dans lequel les snapshots seront enregistrés, du type agent_folder/snapshots
    """
    optimalpolicy = ctxobj['optimalpolicy']
    # modelkey est la clé permettant d'accéder à la configuration du modèle utilisé pour décrire l'environnement
    modelkey = ctxobj['modelkey']
    sub = f'{storage}/{agent_name.replace(".h5","")}'
    if not os.path.isdir(sub):
        os.mkdir(sub)

    for ts in COLD:
        sandbox.play(silent=True, ts=ts, snapshot=True)
        plt.savefig(f'{sub}/{ts}_{modelkey}_{optimalpolicy}')
        plt.close()

    # reconstruit le fichier markdown en inspectant le répertoire snapshot
    with open(f'{sub}/README.md', "w", encoding="utf-8") as readme:
        readme.write(f'# {agent_name.replace(".h5","")}\n')
        all_files = os.listdir(sub)
        all_files.sort()
        family = defaultdict(lambda:[])
        for file in all_files:
            if file.endswith(".png"):
                root = True
                for policy in OPTIMAL_POLICIES :
                    if file.replace(".png","").endswith(policy) :
                        family[policy].append(f'![]({file})')
                        root = False
                if root:
                    readme.write(f'![]({file})\n')
        for key, value in family.items():
            readme.write(f'\n# {key}\n')
            readme.write("\n".join(value))


@click.group()
@click.option('--text', type=int, default=1, prompt='numéro du flux de température extérieure ?')
@click.option('--model', type=click.Choice(MODELS), prompt='modèle ?')
@click.option('--powerlimit', type=float, default=1, prompt='coefficient de réduction de la puissance max ?')
@click.option('--tc', type=int, default=20, prompt='température de consigne en °C')
@click.option('--nbepisodes', type=int, prompt='nombre d\'épisodes à jouer, 0 = mode snapshot : le système joue des épisodes prédéfinis')
@click.option('--optimalpolicy', type=click.Choice(OPTIMAL_POLICIES), prompt='politique du modèle ?')
@click.option('--hystpath', type=str, default=None)
@click.pass_context
def main(ctx, text, model, powerlimit, tc, nbepisodes, optimalpolicy, hystpath):
    ctx.ensure_object(dict)
    ctx.obj['tc'] = tc
    ctx.obj['nbepisodes'] = nbepisodes
    ctx.obj['optimalpolicy'] = optimalpolicy
    ctx.obj['occupation_agent_path'] = None
    if hystpath is not None :
        _, saved = pick_name(name=hystpath)
        if saved :
            ctx.obj['occupation_agent_path'] = hystpath
    ctx.obj['modelkey'] = model
    ctx.obj['model'] = MODELS[model]
    ctx.obj['max_power'] = powerlimit * MAX_POWER

    global circuit
    circuit["Text"] = text

@main.command()
@click.option('--holiday', type=int, default=0, prompt='nombre de jours fériés à intégrer dans les replay')
@click.option('--silent', type=bool, prompt='silent mode = sans montrer les replays des épisodes ?')
@click.option('--k', type=float, default=0.9, prompt='paramètre énergie')
@click.pass_context
def play(ctx, holiday, silent, k):
    print(ctx.obj)
    tc = ctx.obj['tc']
    nbepisodes = ctx.obj['nbepisodes']
    optimalpolicy = ctx.obj['optimalpolicy']
    occupation_agent_path = ctx.obj['occupation_agent_path']
    print(holiday, silent, tc, nbepisodes, optimalpolicy, k)
    input("press")
    if holiday:
        days = [0, 4] if holiday == 1 else [0, 1, 2, 3, 4]
        holidays = []
        for i in range(holiday):
            tirage = random.choice(days)
            if tirage not in holidays:
                holidays.append(tirage)
        for i in holidays:
            circuit["schedule"][i] = [-1,-1]

    # demande à l'utilisateur un nom de réseau
    agent_path, saved = pick_name()

    if saved == True:
        agent, env = load(agent_path, ctx.obj, silent=silent)
        args = {}
        if nbepisodes:
            args["N"] = nbepisodes
        args["k"] = k
        if optimalpolicy == "occupation_permanente":
            sandbox = EvalHyst(agent_path, env, agent, **args)
        elif optimalpolicy == "intermittence":
            sandbox = EvalVote(agent_path, env, agent, **args)
        if occupation_agent_path is not None:
            sandbox.set_occupancy_agent(load_agent(occupation_agent_path))
        if not nbepisodes:
            agent_name = agent_path[agent_path.rfind("/")+1:]
            agent_folder = agent_path[:agent_path.rfind("/")]
            storage = f'{agent_folder}/snapshots'
            if not os.path.isdir(storage):
                os.mkdir(storage)
            snapshots(storage, agent_name, ctx.obj, sandbox)
        else:
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
