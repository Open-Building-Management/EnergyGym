"""a basic sandbox to play with Heatgym"""
import signal
import random
import sys
import click
import numpy as np
import tensorflow as tf
import energy_gym
from energy_gym import get_feed, biosAgenda, pick_name, play_hystnvacancy
from energy_gym import set_extra_params, load
# on importe les configurations existantes de modèles depuis le fichier conf
import conf
from conf import MODELS
from conf import PATH, SCHEDULE, MAX_POWER, TEXT_FEED, REDUCE

INTERVAL = 900
AGENT_TYPES = ["random", "deterministic", "stochastic"]
SIZES = {"weekend": 63 * 3600 // INTERVAL, "week" : 8*24*3600 // INTERVAL}
SCENARIOS = ["Hyst", "Reduce", "Vacancy", "Building", "D2Vacancy"]

# pylint: disable=no-value-for-parameter

def mirror_play(bat):
    """
    Suppose de connaître le nombre de pas pendant lequel le chauffage peut-être arrêté

    A utiliser après avoir fait jouer une période de non-occupation à un modèle

    Rejoue la même période :
    - en arrêtant de chauffer pendant un nombre de pas égal à bat.tot_eko,
    - puis en chauffant de manière continue à partir de bat.tot_eko
    """
    limit = int(bat.tot_eko)
    ts = bat.tsvrai
    tint0 = bat.tint[0]
    tc_episode = bat.tc_episode
    wsize = bat.wsize
    bat.reset(ts=ts, tint=tint0, tc_episode=tc_episode, wsize=wsize)
    while True:
        action = 0 if bat.i < limit else bat.action_space.n - 1
        _, _, done, _ = bat.step(action)
        if done:
            print("MIRROR PLAY")
            peko = stats(bat)
            label = f'{peko:.2f}% d\'énergie économisée'
            label = f'{label} - Tint à l\'ouverture {bat.tint[-1]:.2f}°C'
            bat.render(stepbystep=False, label=label)
            break


def stats(bat):
    """some basic stats"""
    tint_min = np.amin(bat.tint[:-1])
    tint_max = np.amax(bat.tint[:-1])
    tint_moy = np.mean(bat.tint[:-1])
    text_min = np.amin(bat.text[bat.pos:bat.pos+bat.wsize])
    text_max = np.amax(bat.text[bat.pos:bat.pos+bat.wsize])
    text_moy = np.mean(bat.text[bat.pos:bat.pos+bat.wsize])
    print(f'Text min {text_min:.2f} Text moy {text_moy:.2f} Text max {text_max:.2f}')
    print(f'Tint min {tint_min:.2f} Tint moy {tint_moy:.2f} Tint max {tint_max:.2f}')
    peko = 100 * bat.tot_eko / bat.wsize
    print(f'pas de chauffage pendant {bat.tot_eko:.2f} pas')
    print(f'{peko:.2f}% d\'énergie économisée')
    if type(bat).__name__ == "Vacancy":
        print(f'valeur de Tint à l\'ouverture : {bat.tint[-1]:.2f}')
        pmin_eko = 100 * bat.min_eko / bat.wsize
        message = "économie si maintien tc durant épisode"
        message = f'{message} {pmin_eko:.2f}'
        print(message)
    print("***********************************************************")
    return peko


def sig_handler(signum, frame):  # pylint: disable=unused-argument
    """Réception du signal de fermeture"""
    print(f'signal de fermeture ({signum}) reçu')
    sys.exit(0)

NAMES = [*conf.NAMES, "synth_static"]

@click.command()
@click.option('--agent_type', type=click.Choice(AGENT_TYPES), prompt='comportement de l\'agent ?')
@click.option('--random_ts', type=bool, default=False, prompt='timestamp de démarrage aléatoire ?')
@click.option('--scenario', type=click.Choice(SCENARIOS), prompt='scénario ou mode de jeu ?')
@click.option('--size', type=click.Choice(SIZES), prompt='longueur des épisodes ?')
@click.option('--modelkey', type=click.Choice(NAMES), prompt='modèle ?')
@click.option('--stepbystep', type=bool, default=False, prompt='jouer l\'épisode pas à pas ?')
@click.option('--mirrorplay', type=bool, default=False, prompt='jouer le mirror play après avoir joué l\'épisode ?')
@click.option('--tc', type=int, default=20, prompt='consigne moyenne de confort en °C ?')
@click.option('--halfrange', type=int, default=0, prompt='demi-étendue en °C pour travailler à consigne variable ?')
@click.option('--power_factor', type=float, default=1)
@click.option('--mean_prev', type=bool, default=False)
@click.option('--k', type=float, default=1)
@click.option('--k_step', type=float, default=1)
@click.option('--p_c', type=int, default=15)
@click.option('--vote_interval', type=int, nargs=2, default=(-1, 1))
@click.option('--nbh', type=int, default=None)
@click.option('--nbh_forecast', type=int, default=None)
@click.option('--action_space', type=int, default=2)
@click.option('--autosize_max_power', type=bool, default=False)
@click.option('--rc_min', type=int, default=50)
@click.option('--rc_max', type=int, default=100)
def main(agent_type, random_ts, scenario, size, modelkey,
         stepbystep, mirrorplay, tc, halfrange, power_factor, mean_prev,
         k, k_step, p_c, vote_interval, nbh, nbh_forecast, action_space,
         autosize_max_power, rc_min, rc_max):
    """main command"""
    defmodel = conf.generate(bank_name=modelkey, rc_min=rc_min, rc_max=rc_max)
    model = MODELS.get(modelkey, defmodel)
    wsize = SIZES[size]
    model = set_extra_params(model, action_space=action_space, mean_prev=mean_prev)
    model = set_extra_params(model, k=k, k_step=k_step, p_c=p_c, vote_interval=vote_interval)
    model = set_extra_params(model, nbh_forecast=nbh_forecast, nbh=nbh)
    model = set_extra_params(model, autosize_max_power=autosize_max_power)

    text = get_feed(TEXT_FEED, INTERVAL, path=PATH)
    bat = getattr(energy_gym, scenario)(text, power_factor * MAX_POWER, tc, **model)

    # définition de l'agenda d'occupation
    if size == "week" or scenario == "Building":
        agenda = biosAgenda(text.shape[0], INTERVAL, text.start, [], schedule=SCHEDULE)
        bat.set_agenda(agenda)
    agenda = None
    if "Vacancy" in scenario:
        agenda = np.zeros(wsize+1)
        agenda[wsize] = 1
    if scenario == "Hyst":
        agenda = np.ones(wsize+1)

    # réduit hors occupation
    if scenario == "Reduce":
        bat.set_reduce(REDUCE)

    # demande à l'utilisateur un nom de réseau
    if agent_type != "random":
        agent_path, saved = pick_name()
        if not saved :
            sys.exit(0)
        agent = load(agent_path)

    ts = None if random_ts else 1609104740
    nbepisodes = 200 if random_ts else 1
    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)
    for _ in range(nbepisodes):
        tc_episode = tc + random.randint(-halfrange, halfrange)
        if modelkey not in MODELS and modelkey != "synth_static":
            newmodel = conf.generate(bank_name=modelkey, rc_min=rc_min, rc_max=rc_max)
            bat.update_model(newmodel)
        conf.output_model(bat.model)
        state = bat.reset(ts=ts, wsize=wsize, tc_episode=tc_episode)
        rewardtot = 0
        while True :
            if stepbystep:
                bat.render()
            if agent_type == "random":
                # random action
                action = bat.action_space.sample()
            else :
                # using the agent
                result = agent(state.reshape(1, *state.shape))
                if agent_type == "deterministic":
                    # deterministic policy
                    action = np.argmax(result)
                if agent_type == "stochastic":
                    # stochastic policy
                    act_probs = tf.nn.softmax(result, axis=1)
                    action = np.random.choice(act_probs.shape[1], p=act_probs.numpy()[0])
            state, reward, done, _ = bat.step(action)
            rewardtot += reward
            if done:
                if "Vacancy" in scenario:
                    print(f'récompense à l\'arrivée {reward:.2f}')
                print(f'récompense cumulée {rewardtot:.2f}')
                #print(f'{bat.tot_reward}')
                peko = stats(bat)
                if not stepbystep:
                    # pour les scénarios de type Hyst/Vacancy,
                    # on a des custom agendas
                    # qui ne sont pas de la taille de text
                    # on précise donc l'agenda en paramètre
                    optimal_solution = play_hystnvacancy(bat, bat.pos, bat.wsize,
                                                         bat.tint[0], bat.tc_episode, 1,
                                                         agenda=agenda)
                    model_eko = (1 - np.mean(optimal_solution[:,0])) * 100
                    label = f'EKO - modèle : {model_eko:.2f}% - agent : {peko:.2f}%'
                    if "Vacancy" in scenario:
                        label = f'{label} Tint ouverture {bat.tint[-1]:.2f}°C'
                        pmin_eko = 100 * bat.min_eko / bat.wsize
                        label = f'{label}\n EKO MAINTIEN TC PDT EPISODE : {pmin_eko:.2f}%'
                    label = f'{label} R={bat.model["R"]:.2e}K/W C={bat.model["C"]:.2e}J/K'
                    max_power = round(bat.max_power * 1e-3)
                    label = f'{label} MAX_POWER={max_power}kW'
                    bat.render(stepbystep=False, label=label, extra_datas=optimal_solution)
                    if "Vacancy" in scenario and mirrorplay:
                        mirror_play(bat)
                break

    bat.close()


if __name__ == "__main__":
    main()
