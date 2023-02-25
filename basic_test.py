"""les basiques de energy_gym"""
import click

import energy_gym

from conf import PATH, SCHEDULE, MAX_POWER, TEXT_FEED
INTERVAL = 1800

MODES = ["Hyst", "Vacancy"]


@click.command()
@click.option('--mode', type=click.Choice(MODES), prompt='mode ?')
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
    env = getattr(energy_gym, mode)(text, MAX_POWER, 20)
    # on intègre l'agenda
    env.set_agenda(agenda)

    print(env.model)


main()  # pylint: disable=no-value-for-parameter
