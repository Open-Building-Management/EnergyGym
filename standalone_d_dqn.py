"""double dqn"""
import random
import datetime as dt
import math
import click
import numpy as np
import tensorflow as tf
from tensorflow import keras


# on importe les configurations existantes de modèles depuis le fichier conf
from conf import MODELS
from energy_gym import Vacancy, get_feed

# pylint: disable=no-value-for-parameter

GAME = "Heat"
DIR = "TensorBoard/DDQN"
STORE_PATH = f'{DIR}/{GAME}'
MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.0003
GAMMA = 0.97
BATCH_SIZE = 50
TAU = 0.05

NUM_EPISODES = 5400
RENDER = False
NOW = dt.datetime.now().strftime('%d%m%Y%H%M')
DOUBLE_Q = True
CW = 1162.5 #Wh/m3/K
MAX_POWER = 5 * CW * 15


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
    return np.argmax(primary_network(state.reshape(1, -1)))


def train(primary_network, mem, state_size, target_network=None):
    """Generic Network Trainer
    DQN (target_network=None) or DDQN mode"""
    if mem.num_samples < BATCH_SIZE * 3:
        return 0
    batch = mem.sample(BATCH_SIZE)
    states = np.array([val[0] for val in batch])
    actions = np.array([val[1] for val in batch])
    next_states = np.array([(np.zeros(state_size)
                             if val[3] is None else val[3]) for val in batch])
    # predict q values for states
    prim_qsa = primary_network(states)
    # predict q values for next_states
    prim_qsad = primary_network(next_states)
    # updates contient les discounted rewards
    updates = np.array([val[2] for val in batch], dtype=float)
    valid_idxs = np.array(next_states).sum(axis=1) != 0
    batch_idxs = np.arange(BATCH_SIZE)
    if target_network is None:
        # classic DQN
        updates[valid_idxs] += GAMMA * np.amax(prim_qsad.numpy()[valid_idxs, :],
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
        updates[valid_idxs] += GAMMA * q_from_target.numpy()[
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
@click.option('--modelkey', type=click.Choice(MODELS), prompt='modèle ?')
@click.option('--k', type=float, default=0.9, prompt='paramètre énergie')
def main(nbtext, modelkey, k):
    """main command"""
    text = get_feed(nbtext, 3600, "./datas")
    env = Vacancy(text, MAX_POWER, 20, k, **MODELS[modelkey])
    print(env.model)
    state_size = env.observation_space.shape[0]
    num_actions = env.action_space.n

    primary_network = keras.Sequential([
        keras.layers.Dense(50, activation='relu'),
        keras.layers.Dense(num_actions)
    ])

    target_network = keras.Sequential([
        keras.layers.Dense(50, activation='relu'),
        keras.layers.Dense(num_actions)
    ])

    primary_network.compile(optimizer=keras.optimizers.Adam(), loss='mse')
    eps = MAX_EPSILON
    steps = 0

    memory = Memory(50000)

    def dot(num):
        """replace dots"""
        return str(num).replace(".", "dot")

    for i in range(NUM_EPISODES):
        state = env.reset()
        cnt = 0
        avg_loss = 0
        while True:
            if RENDER:
                env.render()
            action = choose_action(state, primary_network, eps, num_actions)
            next_state, reward, done, _ = env.step(action)
            if i == 0 and env.i == 1:
                # première étape du premier épisode : reward_label existe
                suffix = f'{modelkey}_k={dot(k)}_GAMMA={dot(GAMMA)}_{env.reward_label}'
                train_writer = tf.summary.create_file_writer(STORE_PATH + f"/{NOW}_{suffix}")

            if done:
                next_state = None
            # store in memory
            memory.add_sample((state, action, reward, next_state))

            loss = train(primary_network, memory, state_size, target_network if DOUBLE_Q else None)
            avg_loss += loss

            state = next_state

            # exponentially decay the eps value
            steps += 1
            eps = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * steps)

            if done:
                avg_loss /= cnt
                message = f'Episode: {i}, Reward: {reward:.3f}'
                message = f'{message}, avg loss: {avg_loss:.3f}, eps: {eps:.3f}'
                print(message)
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
                print(env.tint[-2:])
                peko = (env.tot_eko * 100) // env.wsize
                print(f'{peko}% d\'énergie économisée')
                print("***********************************************************")
                with train_writer.as_default():
                    tf.summary.scalar('reward', reward, step=i)
                    tf.summary.scalar('avg loss', avg_loss, step=i)
                    #tf.summary.scalar('Text/min', text_min, step=i)
                    #tf.summary.scalar('Text/max', text_max, step=i)
                break

            cnt += 1

    save = input("save ? Y=yes")
    if save == "Y":
        primary_network.save(f'{DIR}/{GAME}_aimlDDQN_{NOW}_{suffix}')


if __name__ == "__main__":
    main()
