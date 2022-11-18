import numpy as np
from EnergyGym import getTruth, getFeed, pickName, Building, Vacancy
import signal

interval = 3600
wsize = 1 + 8*24*3600//interval
dir = "datas"
schedule = np.array([ [7,17], [7,17], [7,17], [7,17], [7,17], [-1,-1], [-1,-1] ])
Cw = 1162.5 #Wh/m3/K
# debit de 5m3/h et deltaT entre départ et retour de 15°C
max_power = 5 * Cw * 15
hh = 1
circuit = {"Text":1, "dir": dir, "schedule": schedule, "interval": interval, "wsize": wsize}

def load(agent_path):
    import tensorflow as tf
    agent = tf.keras.models.load_model(agent_path, compile = False, custom_objects={'Functional':tf.keras.models.Model})
    return agent

def mirrorPlay(bat):
    """
    Suppose de connaître le nombre de pas pendant lequel le chauffage peut-être arrêté : limit

    A utiliser après avoir fait jouer une période de non-occupation à un modèle

    Rejoue la même période :
    - en arrêtant de chauffer pendant un nombre de pas égal à limit,
    - puis en chauffant de manière continue à partir de limit
    """
    limit = bat.tot_eko
    ts = bat._tsvrai
    state = bat.reset(ts = ts)
    while True:
        action = 0 if bat.i < limit else 1
        state, reward, done, _ = bat.step(action)
        if done:
            print("MIRROR PLAY")
            stats(bat)
            label = "chauffage arrêté pendant {} pas".format(np.sum(bat.tot_eko))
            label = "{} - Tint à l'ouverture {:.2f}°C".format(label, bat.tint[-2])
            bat.render(stepbystep = False, label = label)
            break

def stats(bat):
    Tint_min = np.amin(bat.tint[:-1])
    Tint_max = np.amax(bat.tint[:-1])
    Tint_moy = np.mean(bat.tint[:-1])
    Text_min = np.amin(bat.text[bat.pos:bat.pos+bat.wsize])
    Text_max = np.amax(bat.text[bat.pos:bat.pos+bat.wsize])
    Text_moy = np.mean(bat.text[bat.pos:bat.pos+bat.wsize])
    print("Text min {:.2f} Text moy {:.2f} Text max {:.2f}".format(Text_min, Text_moy, Text_max))
    print("Tint min {:.2f} Tint moy {:.2f} Tint max {:.2f}".format(Tint_min, Tint_moy, Tint_max))
    if bat.label == "vacancy":
        print("valeur de Tint à l'ouverture : {:.2f}".format(bat.tint[-2]))
        peko = (bat.tot_eko * 100) // bat.wsize
        print("pas de chauffage pendant {} pas".format(bat.tot_eko))
        print("{}% d\'énergie économisée".format(peko))
    print("***********************************************************")

def _sig_handler(signum, frame):
    """
    Réception du signal de fermeture
    """
    print(f'signal de fermeture ({signum}) reçu')
    exit(0)

# on importe les configurations existantes de modèles depuis le fichier conf
from conf import models
agentTypes = ["random", "deterministic", "stochastic"]
modes = ["vacancy", "week"]
import click
@click.command()
@click.option('--agent_type', type=click.Choice(agentTypes), prompt='comportement de l\'agent ?')
@click.option('--random_ts', type=bool, default=False, prompt='timestamp de démarrage aléatoire ?')
@click.option('--mode', type=click.Choice(modes), prompt='type d\'épisode : période de non-occupation, semaine ?')
@click.option('--model', type=click.Choice(models), prompt='modèle ?')
@click.option('--stepbystep', type=bool, default=False, prompt='jouer l\'épisode pas à pas ?')
def main(agent_type, random_ts, mode, model, stepbystep):
    R = models[model]["R"]
    C = models[model]["C"]
    if mode == "week":
        Text, agenda = getTruth(circuit, visualCheck=False)
        bat = Building(Text, agenda, wsize, max_power, 20, 0.9, R=R, C=C)
    if mode == "vacancy":
        Text = getFeed(circuit["Text"], circuit["interval"])
        bat = Vacancy(Text, max_power, 20, 0.9, R=R, C=C)

    # demande à l'utilisateur un nom de réseau
    if agent_type != "random":
        agent_path, saved = pickName()
        if not saved :
            import sys
            sys.exit(0)
        agent = load(agent_path)

    ts = None if random_ts else 1609104740
    nbepisodes = 200 if random_ts else 1
    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)
    for episode in range(nbepisodes):
        state = bat.reset(ts = ts)
        rewardTot = 0
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
            rewardTot += reward
            if done:
                if mode == "vacancy":
                    print("récompense à l'arrivée {}".format(reward))
                print("récompense cumulée {}".format(rewardTot))
                stats(bat)
                if not stepbystep:
                    label = None
                    if mode == "vacancy":
                        label = "chauffage arrêté pendant {} pas".format(bat.tot_eko)
                        label = "{} - Tint à l'ouverture {:.2f}°C".format(label, bat.tint[-2])
                    bat.render(stepbystep = False, label = label)
                    if mode == "vacancy":
                        mirrorPlay(bat)
                break

    bat.close()

if __name__ == "__main__":
    main()
