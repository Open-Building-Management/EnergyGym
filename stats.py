"""joue des épisodes d'une semaine et produit des stats"""
import signal
import click
import energy_gym
from energy_gym import EvaluateGym
from energy_gym import get_feed, biosAgenda, pick_name, set_extra_params
from energy_gym import load, freeze
import conf
from conf import MODELS
from conf import PATH, SCHEDULE, MAX_POWER, TEXT_FEED

INTERVAL = 3600
WSIZE = 8*24*3600 // INTERVAL

NAMES = [*MODELS.keys(), "synth"]

def sig_handler(signum, frame):  # pylint: disable=unused-argument
    """gracefull shutdown"""
    print(f'signal de fermeture reçu {signum}')
    raise SystemExit


def create_sandbox(scenario, text, agenda, model, agent_path, nb_episodes):
    """build the evaluation sandbox"""
    bat = getattr(energy_gym, scenario)(text, MAX_POWER, 20, **model)
    bat.set_agenda(agenda)
    print(bat.model)
    agent = load(agent_path)
    return EvaluateGym(agent_path, bat, agent, N=nb_episodes)


# pylint: disable=no-value-for-parameter
@click.command()
@click.option('--modelkey', type=click.Choice(NAMES), prompt='modèle ?')
@click.option('--nbh', type=int, default=None)
@click.option('--nbh_forecast', type=int, default=None)
@click.option('--mean_prev', type=bool, default=False)
@click.option('--generate_stats', type=bool, default=True, prompt='generer les stats ?')
@click.option('--nb_off', type=int, default=0, prompt='nbr jours fériés à intégrer ?')
@click.option('--action_space', type=int, default=2)
@click.option('--autosize_max_power', type=bool, default=False)
@click.option('--newmodel_at_each_episode', type=bool, default=False)
@click.option('--nb_episodes', type=int, default=900)
@click.option('--rc_min', type=int, default=50)
@click.option('--rc_max', type=int, default=100)
def main(modelkey, nbh, nbh_forecast, mean_prev, generate_stats, nb_off,
         action_space, autosize_max_power, newmodel_at_each_episode,
         nb_episodes, rc_min, rc_max):
    """main command"""
    defmodel = conf.generate(bank_name=modelkey, rc_min=rc_min, rc_max=rc_max)
    model = MODELS.get(modelkey, defmodel)
    model = set_extra_params(model, action_space=action_space)
    model = set_extra_params(model, mean_prev=mean_prev)
    model = set_extra_params(model, nbh_forecast=nbh_forecast, nbh=nbh)
    model = set_extra_params(model, autosize_max_power=autosize_max_power)
    text = get_feed(TEXT_FEED, INTERVAL, path=PATH)
    # demande à l'utilisateur des chemins de réseaux
    agent_path, agent_exists = pick_name()
    question = "chemin agent hystérésis pour les périodes d'occupation ?"
    hyst_path, hyst_exists = pick_name(question=question)
    question = "agent avec qui comparer pas à pas ?"
    concurrent_path, concurrent_exists = pick_name(question=question)
    if agent_exists:
        scenario = "Hyst" if "Hyst" in agent_path else "Building"
        # modification de SCHEDULE pour intégrer des jours chomés
        if nb_off:
            for i in freeze(nb_off):
                SCHEDULE[i] = [-1, -1]
        # génération de l'agenda
        agenda = biosAgenda(text.shape[0], INTERVAL, text.start, [], schedule=SCHEDULE)
        agent_box = create_sandbox(scenario, text, agenda,
                                   model, agent_path,
                                   nb_episodes)
        if hyst_exists:
            hyst = load(hyst_path)
            agent_box.set_occupancy_agent(hyst)
        # on passe en paramètre la taille de l'épisode
        # nécessaire si on veut jouer un hystérésis sur toute une semaine
        #for fix_tc in [False, True]:
        #    agent_box.play_gym(ts=1605821540, wsize=WSIZE, fix_tc=fix_tc)
        if concurrent_exists:
            concurrent_model = model
            concurrent_model["nbh"] = 0
            concurrent_box = create_sandbox(scenario, text, agenda,
                                            concurrent_model, concurrent_path,
                                            nb_episodes)
            if hyst_exists:
                concurrent_box.set_occupancy_agent(hyst)

        signal.signal(signal.SIGINT, sig_handler)
        signal.signal(signal.SIGTERM, sig_handler)

        for _ in range(nb_episodes):
            if newmodel_at_each_episode:
                newmodel = conf.generate(bank_name=modelkey, rc_min=rc_min, rc_max=rc_max)
                print(newmodel)
                agent_box.update_model(newmodel)
                if concurrent_exists:
                    concurrent_box.update_model(newmodel)
            if not generate_stats:
                agent_box.play_gym(wsize=WSIZE)
            else:
                agent_box.play_base(wsize=WSIZE)
            agent_box.nb_episode += 1
            if concurrent_exists:
                tint0, ts = agent_box.get_episode_params()
                if not generate_stats:
                    concurrent_box.play_gym(wsize=WSIZE, ts=ts, tint=tint0)
                else:
                    concurrent_box.play_base(wsize=WSIZE, ts=ts, tint=tint0)
                concurrent_box.nb_episode += 1

        # close and record stats
        agent_box.close(suffix=modelkey, random_model=newmodel_at_each_episode)
        if concurrent_exists:
            concurrent_box.close(suffix=modelkey, random_model=newmodel_at_each_episode)

if __name__ == "__main__":
    main()
