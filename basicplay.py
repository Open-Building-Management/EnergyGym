"""a basic sandbox to play with Heatgym"""
import signal
import sys
import click
import numpy as np
import tensorflow as tf
from energy_gym import get_truth, get_feed, pick_name, Building, Vacancy
# on importe les configurations existantes de modèles depuis le fichier conf
from conf import MODELS

AGENT_TYPES = ["random", "deterministic", "stochastic"]
MODES = ["vacancy", "week"]

# pylint: disable=no-value-for-parameter

INTERVAL = 3600
WSIZE = 1 + 8*24*3600 // INTERVAL
PATH = "datas"
SCHEDULE = np.array([[7, 17], [7, 17], [7, 17], [7, 17], [7, 17], [-1, -1], [-1, -1]])
CW = 1162.5 #Wh/m3/K
# debit de 5m3/h et deltaT entre départ et retour de 15°C
MAX_POWER = 5 * CW * 15
CIRCUIT = {"Text":1, "dir": PATH, "schedule": SCHEDULE, "interval": INTERVAL, "wsize": WSIZE}

def load(agent_path):
    """load tensorflow network"""
    # custom_objects est nécessaire pour charger certains réseaux entrainés sur le cloud, via les github actions
    agent = tf.keras.models.load_model(agent_path, compile=False, custom_objects={'Functional':tf.keras.models.Model})
    return agent

def mirror_play(bat):
    """
    Suppose de connaître le nombre de pas pendant lequel le chauffage peut-être arrêté

    A utiliser après avoir fait jouer une période de non-occupation à un modèle

    Rejoue la même période :
    - en arrêtant de chauffer pendant un nombre de pas égal à bat.tot_eko,
    - puis en chauffant de manière continue à partir de bat.tot_eko
    """
    limit = bat.tot_eko
    ts = bat.tsvrai
    bat.reset(ts=ts)
    while True:
        action = 0 if bat.i < limit else 1
        _, _, done, _ = bat.step(action)
        if done:
            print("MIRROR PLAY")
            stats(bat)
            label = f'chauffage arrêté pendant {np.sum(bat.tot_eko)} pas'
            label = f'{label} - Tint à l\'ouverture {bat.tint[-2]:.2f}°C'
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
    if bat.label == "vacancy":
        print(f'valeur de Tint à l\'ouverture : {bat.tint[-2]:.2f}')
        peko = (bat.tot_eko * 100) // bat.wsize
        print(f'pas de chauffage pendant {bat.tot_eko} pas')
        print(f'{peko}% d\'énergie économisée')
    print("***********************************************************")

def sig_handler(signum, frame):  # pylint: disable=unused-argument
    """Réception du signal de fermeture"""
    print(f'signal de fermeture ({signum}) reçu')
    sys.exit(0)

@click.command()
@click.option('--agent_type', type=click.Choice(AGENT_TYPES), prompt='comportement de l\'agent ?')
@click.option('--random_ts', type=bool, default=False, prompt='timestamp de démarrage aléatoire ?')
@click.option('--mode', type=click.Choice(MODES), prompt='type d\'épisode : période de non-occupation, semaine ?')
@click.option('--model', type=click.Choice(MODELS), prompt='modèle ?')
@click.option('--stepbystep', type=bool, default=False, prompt='jouer l\'épisode pas à pas ?')
def main(agent_type, random_ts, mode, model, stepbystep):
    """main command"""
    R = MODELS[model]["R"]
    C = MODELS[model]["C"]
    if mode == "week":
        text, agenda = get_truth(CIRCUIT, visual_check=False)
        bat = Building(text, agenda, WSIZE, MAX_POWER, 20, 0.9, R=R, C=C)
    if mode == "vacancy":
        text = get_feed(CIRCUIT["Text"], CIRCUIT["interval"])
        bat = Vacancy(text, MAX_POWER, 20, 0.9, R=R, C=C)

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
        state = bat.reset(ts=ts)
        rewardtot = 0
        while True :
            if stepbystep:
                bat.render()
            if agent_type == "random":
                # random action
                action = bat.action_space.sample()
            else :
                # using the agent
                # on peut passer en argument state.reshape((1, -1))
                result = agent(state.reshape(1, state.shape[0]))
                if agent_type == "deterministic":
                    # deterministic policy
                    action = np.argmax(result)
                if agent_type == "stochastic":
                    # stochastic policy
                    import tensorflow as tf
                    act_probs = tf.nn.softmax(result, axis=1)
                    action = np.random.choice(act_probs.shape[1], p=act_probs.numpy()[0])
            state, reward, done, _ = bat.step(action)
            rewardtot += reward
            if done:
                if mode == "vacancy":
                    print(f'récompense à l\'arrivée {reward:.2f}')
                print(f'récompense cumulée {rewardtot:.2f}')
                stats(bat)
                if not stepbystep:
                    label = None
                    if mode == "vacancy":
                        label = f'chauffage arrêté pendant {bat.tot_eko} pas'
                        label = f'{label} - Tint à l\'ouverture {bat.tint[-2]:.2f}°C'
                    bat.render(stepbystep=False, label=label)
                    if mode == "vacancy":
                        mirror_play(bat)
                break

    bat.close()


if __name__ == "__main__":
    main()
