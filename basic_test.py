"""les basiques de energy_gym"""
import numpy as np
import click

import energy_gym

INTERVAL = 1800
# nombre d'intervalles sur lequel la simulation sera menée
WSIZE = 1 + 8*24*3600//INTERVAL
PATH = "datas"

SCHEDULE = np.array([[7, 17], [7, 17], [7, 17], [7, 17], [7, 17], [-1, -1], [-1, -1]])
CW = 1162.5 #Wh/m3/K
MAX_POWER = 5 * CW * 15

TEXT_FEED = 1

MODES = ["Hyst", "Vacancy"]


@click.command()
@click.option('--mode', type=click.Choice(MODES), prompt='hystérésis, non-occupation avec température à atteindre à la fin ?')
def main(mode):
    """main command"""
    text = energy_gym.get_feed(TEXT_FEED, INTERVAL, path=PATH)
    agenda = energy_gym.biosAgenda(text.shape[0], INTERVAL, text.start, [], schedule=SCHEDULE)

    # pour accéder à des fonctions qui ne sont pas listées dans le init
    i = 0
    nbh = energy_gym.planning.get_level_duration(agenda, i) * INTERVAL / 3600
    print(f'position {i}, changement d\'occupation dans {nbh} heures' )

    # l'utilisateur choisit le mode parmi une liste
    # ET une seule instruction suffit pour construire l'environnement
    env = getattr(energy_gym, mode)(text, MAX_POWER, 20, 0.9)
    # on intègre l'agenda
    env.set_agenda(agenda)

    print(env.model)


main()  # pylint: disable=no-value-for-parameter
