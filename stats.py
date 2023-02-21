"""joue des épisodes d'une semaine et produit des stats"""
import click
import numpy as np
import energy_gym
from energy_gym import Evaluate
from energy_gym import get_feed, biosAgenda, pick_name, play_hystnvacancy
from conf import MODELS
from basicplay import load
from standalone_d_dqn import set_extra_params

INTERVAL = 3600
WSIZE = 8*24*3600 // INTERVAL
PATH = "datas"
SCHEDULE = np.array([[7, 17], [7, 17], [7, 17], [7, 17], [7, 17], [-1, -1], [-1, -1]])
CW = 1162.5 #Wh/m3/K
# debit de 5m3/h et deltaT entre départ et retour de 15°C
MAX_POWER = 5 * CW * 15
TEXT_FEED = 1

@click.command()
@click.option('--modelkey', type=click.Choice(MODELS), prompt='modèle ?')
@click.option('--nbh', type=int, default=None)
@click.option('--nbh_forecast', type=int, default=None)
def main(modelkey, nbh, nbh_forecast):
    """main command"""
    model = MODELS[modelkey]
    model = set_extra_params(model, nbh_forecast=nbh_forecast, nbh=nbh)
    text = get_feed(TEXT_FEED, INTERVAL, path=PATH)
    # demande à l'utilisateur des chemins de réseaux
    agent_path, saved = pick_name()
    question = "chemin agent hystérésis pour les périodes d'occupation ?"
    hyst_path, saved_hyst = pick_name(question=question)
    if saved:
        scenario = "Hyst" if "Hyst" in agent_path else "Building"
        bat = getattr(energy_gym, scenario)(text, MAX_POWER, 20, **model)
        agenda = biosAgenda(text.shape[0], INTERVAL, text.start, [], schedule=SCHEDULE)
        bat.set_agenda(agenda)
        print(bat.model)
        agent = load(agent_path)
        sandbox = Evaluate(agent_path, bat, agent)
        if saved_hyst:
            hyst = load(hyst_path)
            sandbox.set_occupancy_agent(hyst)
        while True:
            # on passe en paramètre la taille de l'épisode
            # nécessaire si on veut jouer un hystérésis sur toute une semaine
            sandbox.play_gym(silent=False, wsize=WSIZE)

if __name__ == "__main__":
    main()
