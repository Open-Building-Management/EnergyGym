"""joue des épisodes d'une semaine et produit des stats"""
import random
import click
import energy_gym
from energy_gym import Evaluate
from energy_gym import get_feed, biosAgenda, pick_name
from conf import MODELS
from standalone_d_dqn import set_extra_params
from basicplay import load
from basicplay import PATH, SCHEDULE, MAX_POWER, TEXT_FEED

INTERVAL = 3600
WSIZE = 8*24*3600 // INTERVAL

def freeze(nb):
    """retourne le tableau des numéros des jours
    chomés dans la semaine, en plus du week-end
    
    nb : nombre de jours chomés à injecter
    """
    days = [0, 4] if nb == 1 else [0, 1, 2, 3, 4]
    holidays = []
    for i in range(nb):
        tirage = random.choice(days)
        if tirage not in holidays:
            holidays.append(tirage)
    return holidays


# pylint: disable=no-value-for-parameter
@click.command()
@click.option('--modelkey', type=click.Choice(MODELS), prompt='modèle ?')
@click.option('--nbh', type=int, default=None)
@click.option('--nbh_forecast', type=int, default=None)
@click.option('--generate_stats', type=bool, default=True, prompt='generer les stats ?')
@click.option('--nb_off', type=int, default=0, prompt='nbr jours fériés à intégrer ?')
def main(modelkey, nbh, nbh_forecast, generate_stats, nb_off):
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
        # modification de SCHEDULE pour intégrer des jours chomés
        if nb_off:
            for i in freeze(nb_off):
                SCHEDULE[i] = [-1, -1]
        # génération de l'agenda
        agenda = biosAgenda(text.shape[0], INTERVAL, text.start, [], schedule=SCHEDULE)
        bat.set_agenda(agenda)
        print(bat.model)
        agent = load(agent_path)
        sandbox = Evaluate(agent_path, bat, agent)
        if saved_hyst:
            hyst = load(hyst_path)
            sandbox.set_occupancy_agent(hyst)
        # on passe en paramètre la taille de l'épisode
        # nécessaire si on veut jouer un hystérésis sur toute une semaine
        sandbox.run_gym(silent=generate_stats, wsize=WSIZE)
        sandbox.close(suffix=modelkey)

if __name__ == "__main__":
    main()