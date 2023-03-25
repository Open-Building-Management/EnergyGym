"""double dqn"""
import random
import datetime as dt
import math
import click
import numpy as np
import tensorflow as tf
from tensorflow import keras


# on importe les configurations existantes de modèles depuis le fichier conf
import conf
from conf import MODELS
from conf import PATH, MAX_POWER
import energy_gym
from energy_gym import get_feed, set_extra_params

# pylint: disable=no-value-for-parameter

GAME = "Heat"
DIR = "TensorBoard/DDQN"
STORE_PATH = f'{DIR}/{GAME}'
MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.0003
#LAMBDA = 5e-5
BATCH_SIZE = 50
TAU = 0.05

RENDER = False
NOW = dt.datetime.now().strftime('%d%m%Y%H%M')
DOUBLE_Q = True
INTERVAL = 3600

SCENARIOS = ["Hyst",
             "Vacancy", "StepRewardVacancy", "TopLimitVacancy",
             "D2Vacancy"]


def show_episode_stats(env):
    """affiche les statistiques de l'épisode en cours pour l'environnement"""
    message = f'consigne de température intérieure: {env.tc_episode}°C'
    print(message)
    tint_min = np.amin(env.tint)
    tint_max = np.amax(env.tint)
    tint_moy = np.mean(env.tint)
    text_min = np.amin(env.text[env.pos:env.pos+env.wsize])
    text_max = np.amax(env.text[env.pos:env.pos+env.wsize])
    text_moy = np.mean(env.text[env.pos:env.pos+env.wsize])
    message = f'Text min {text_min:.2f} Text moy {text_moy:.2f}'
    message = f'{message} Text max {text_max:.2f}'
    print(message)
    message = f'Tint min {tint_min:.2f} Tint moy {tint_moy:.2f}'
    message = f'{message} Tint max {tint_max:.2f}'
    print(message)
    print(env.tint[-1:])
    peko = 100 * env.tot_eko // env.wsize
    print(f'{peko}% d\'énergie économisée')
    pmin_eko = 100 * env.min_eko // env.wsize
    print(f'économie si maintien tc durant épisode: {pmin_eko:.2f}%')


def add_scalars_to_tensorboard(train_writer, i, reward, avg_loss, env):
    """met à jour les indicateurs qualité tensorboard pour l'épisode i"""
    with train_writer.as_default():
        tf.summary.scalar('reward', reward, step=i)
        tf.summary.scalar('avg loss', avg_loss, step=i)
        delta_to_tc = abs(env.tc_episode - env.tint[-1])
        tf.summary.scalar('respect_tc_ouverture', delta_to_tc, step=i)
        if "Vacancy" in env.__class__.__name__:
            gain = 100 * (env.tot_eko - env.min_eko) // env.wsize
            tf.summary.scalar('gain_sur_baseline', gain, step=i)


class Memory:
    """experience replay memory"""
    def __init__(self, max_memory):
        self._max_memory = max_memory
        self._samples = []

    def add_sample(self, sample):
        """add a sample"""
        self._samples.append(sample)
        if len(self._samples) > self._max_memory:
            self._samples.pop(0)

    def sample(self, no_samples):
        """extract a batch"""
        if no_samples > len(self._samples):
            return random.sample(self._samples, len(self._samples))
        return random.sample(self._samples, no_samples)

    @property
    def num_samples(self):
        """memory size"""
        return len(self._samples)


def choose_action(state, primary_network, eps, num_actions):
    """epsilon greedy action"""
    if random.random() < eps:
        return random.randint(0, num_actions - 1)
    return np.argmax(primary_network(state.reshape(1, *state.shape)))


def train(primary_network, mem, state_shape, gamma, target_network=None):
    """Generic Network Trainer
    DQN (target_network=None) or DDQN mode"""
    if mem.num_samples < BATCH_SIZE * 3:
        return 0
    batch = mem.sample(BATCH_SIZE)
    states = np.array([val[0] for val in batch])
    actions = np.array([val[1] for val in batch])
    next_states = np.array([(np.zeros(state_shape)
                             if val[3] is None else val[3]) for val in batch])
    # predict q values for states
    prim_qsa = primary_network(states)
    # predict q values for next_states
    prim_qsad = primary_network(next_states)
    # updates contient les discounted rewards
    updates = np.array([val[2] for val in batch], dtype=float)
    # les axes des samples
    smp_axis = tuple(range(1, len(next_states.shape)))
    valid_idxs = np.array(next_states).sum(axis=smp_axis) != 0
    batch_idxs = np.arange(BATCH_SIZE)
    if target_network is None:
        # classic DQN
        updates[valid_idxs] += gamma * np.amax(prim_qsad.numpy()[valid_idxs, :],
                                               axis=1)
    else:
        # double DQN
        # 1) prim_actions = indices pour lesquels qsad prend sa valeur max
        # 2) q_from_target = q values for next_states avec le target_network
        # 3) on calcule les discounted rewards à partir des valeurs du target_network
        #    MAIS avec les indices fournis par le primary_network
        # cf google deepmind : https://arxiv.org/pdf/1509.06461.pdf
        prim_actions = np.argmax(prim_qsad.numpy(), axis=1)
        q_from_target = target_network(next_states)
        updates[valid_idxs] += gamma * q_from_target.numpy()[
            batch_idxs[valid_idxs],
            prim_actions[valid_idxs]
        ]

    # on calcule le target_q à utiliser dans train_on_batch
    target_q = prim_qsa.numpy()
    target_q[batch_idxs, actions] = updates

    loss = primary_network.train_on_batch(states, target_q)

    if target_network is not None:
        # slowly update target_network from primary_network
        for t_tv, p_tv in zip(target_network.trainable_variables,
                              primary_network.trainable_variables):
            t_tv.assign(t_tv * (1 - TAU) + p_tv * TAU)
    return loss


@click.command()
@click.option('--nbtext', type=int, default=1, prompt='numéro du flux temp. extérieure ?')
@click.option('--modelkey', type=click.Choice(conf.NAMES), prompt='modèle ou banque ?')
@click.option('--scenario', type=click.Choice(SCENARIOS), default="Vacancy", prompt='scénario ?')
@click.option('--tc', type=int, default=20, prompt='consigne moyenne de confort en °C ?')
@click.option('--halfrange', type=int, default=0, prompt='demi-étendue en °C pour W à consigne variable ?')
@click.option('--gamma', type=float, default=0.97, prompt='discount parameter GAMMA ?')
@click.option('--num_episodes', type=int, default=5400, prompt="nombre d'épisodes ?")
@click.option('--nb_mlp_per_layer', type=int, default=50, prompt="nombre de neurones par couche ?")
@click.option('--mean_prev', type=bool, default=False)
@click.option('--k', type=float, default=1)
@click.option('--k_step', type=float, default=1)
@click.option('--p_c', type=int, default=15)
@click.option('--vote_interval', type=int, nargs=2, default=(-1, 1))
@click.option('--nbh', type=int, default=None)
@click.option('--nbh_forecast', type=int, default=None)
@click.option('--action_space', type=int, default=2)
@click.option('--verbose', type=bool, default=False)
@click.option('--autosize_max_power', type=bool, default=False)
@click.option('--rc_min', type=int, default=50)
@click.option('--rc_max', type=int, default=100)
def main(nbtext, modelkey, scenario, tc, halfrange, gamma, num_episodes,
         nb_mlp_per_layer, mean_prev, k, k_step, p_c, vote_interval,
         nbh, nbh_forecast, action_space, verbose,
         autosize_max_power, rc_min, rc_max):
    """main command"""
    text = get_feed(nbtext, INTERVAL, path=PATH)
    defmodel = conf.generate(bank_name=modelkey)
    model = MODELS.get(modelkey, defmodel)

    model = set_extra_params(model, action_space=action_space)
    model = set_extra_params(model, mean_prev=mean_prev)
    model = set_extra_params(model, k=k, k_step=k_step, p_c=p_c)
    model = set_extra_params(model, vote_interval=vote_interval)
    model = set_extra_params(model, nbh_forecast=nbh_forecast, nbh=nbh)
    model = set_extra_params(model, autosize_max_power=autosize_max_power)

    env = getattr(energy_gym, scenario)(text, MAX_POWER, tc, **model)

    print(env.model)
    input("press a key")
    state_shape = env.observation_space.shape
    num_actions = env.action_space.n

    primary_network = keras.Sequential([
        keras.layers.Dense(nb_mlp_per_layer, activation='relu'),
        keras.layers.Dense(nb_mlp_per_layer, activation='relu'),
        keras.layers.Dense(num_actions)
    ])

    target_network = keras.Sequential([
        keras.layers.Dense(nb_mlp_per_layer, activation='relu'),
        keras.layers.Dense(nb_mlp_per_layer, activation='relu'),
        keras.layers.Dense(num_actions)
    ])

    primary_network.compile(optimizer=keras.optimizers.Adam(), loss='mse')
    eps = MAX_EPSILON
    steps = 0

    memory = Memory(50000)

    for i in range(num_episodes):
        tc_episode = tc + random.randint(-halfrange, halfrange)
        if modelkey not in MODELS:
            newmodel = conf.generate(bank_name=modelkey, rc_min=rc_min, rc_max=rc_max)
            env.update_model(newmodel)
        print("***********************************************************")
        conf.output_model(env.model)
        state = env.reset(tc_episode=tc_episode)
        max_power = round(env.max_power * 1e-3)
        print(f'max power : {max_power} kW')

        cnt = 0
        avg_loss = 0
        while True:
            if RENDER:
                env.render()
            if verbose:
                print(state)
                input("press a key")
            action = choose_action(state, primary_network, eps, num_actions)
            next_state, reward, done, _ = env.step(action)
            if i == 0 and env.i == 1:
                # première étape du premier épisode
                suffix = modelkey
                suffix = f'{suffix}_no_rc_min' if rc_min < 0 else f'{suffix}_rc_min={rc_min}'
                suffix = f'{suffix}_no_rc_max' if rc_max < 0 else f'{suffix}_rc_max={rc_max}'
                suffix = f'{suffix}_GAMMA={gamma:.2e}'
                suffix = f'{suffix}_LAMBDA={LAMBDA:.2e}'
                suffix = f'{suffix}_NBACTIONS={num_actions}'
                if nb_mlp_per_layer != 50:
                    suffix = f'{suffix}_{nb_mlp_per_layer}MLP'
                if autosize_max_power:
                    suffix = f'{suffix}_AUTOPOWER'
                else:
                    suffix = f'{suffix}_{max_power}kW'
                suffix = f'{suffix}_tc={tc}'
                if halfrange:
                    suffix = f'{suffix}+ou-{halfrange}'
                if nbh:
                    suffix = f'{suffix}_past={nbh}h'
                if nbh_forecast:
                    suffix = f'{suffix}_future={nbh_forecast}h'
                if mean_prev:
                    suffix = f'{suffix}_MEAN_PREV'
                if "Vacancy" in scenario:
                    suffix = f'{suffix}_k={k:.2e}_k_step={k_step:.2e}_p_c={p_c}'
                    suffix = f'{suffix}_vote_interval={vote_interval[0]}A{vote_interval[1]}'
                tw_path = f'{STORE_PATH}/{scenario}{num_episodes}_{NOW}_{suffix}'
                train_writer = tf.summary.create_file_writer(tw_path)

            if done:
                next_state = None
            # store in memory
            memory.add_sample((state, action, reward, next_state))

            loss = train(primary_network, memory, state_shape, gamma, target_network if DOUBLE_Q else None)
            avg_loss += loss

            state = next_state

            # exponentially decay the eps value
            steps += 1
            eps = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * steps)

            if done:
                avg_loss /= cnt
                message = f'Episode: {i}, Reward: {reward:.3f}, Total Reward: {env.tot_reward:.3f}'
                message = f'{message}, avg loss: {avg_loss:.3f}, eps: {eps:.3f}'
                print(message)
                show_episode_stats(env)
                add_scalars_to_tensorboard(train_writer, i, reward, avg_loss, env)
                break

            cnt += 1

    save = input("save ? Y=yes")
    if save == "Y":
        primary_network.save(f'{STORE_PATH}_{scenario}{num_episodes}_{NOW}_{suffix}')


if __name__ == "__main__":
    main()
