"""prioritised experience replay with a dueling architecture"""
import random
import datetime as dt
import click
import numpy as np
import tensorflow as tf
from tensorflow import keras

from conf import MODELS
from conf import PATH, MAX_POWER
import energy_gym
from energy_gym import get_feed, set_extra_params
from dueling import linear_decay, update_network, gen_random_model_and_reset
from standalone_d_dqn import show_episode_stats, add_scalars_to_tensorboard
from shared_rl_tools import DQModel, Memory, Batch, get_td_error

# pylint: disable=no-value-for-parameter

GAME = "Heat"
DIR = "TensorBoard/PER_DuelingQ"
STORE_PATH = f'{DIR}/{GAME}'
MAX_EPS = 1
MIN_EPS = 0.1
EPS_ITER = 5000
GAMMA = 0.95
BATCH_SIZE = 50
TAU = 0.08
DELAY_TRAINING = 300
BETA_ITER = 5000
MIN_BETA = 0.4
MAX_BETA = 1.0

INTERVAL = 3600
NOW = dt.datetime.now().strftime('%d%m%Y%H%M')


def choose_action(state, primary_network, eps, step, num_actions):
    """epsilon greedy action"""
    if step < DELAY_TRAINING or random.random() < eps:
        return random.randrange(0, num_actions)
    return np.argmax(primary_network(state.reshape(1, *state.shape)))


def train(primary_network, memory, target_network):
    """train on batch"""
    batch, idxs, is_weights = memory.sample_tree(BATCH_SIZE)
    qt_for_train, errors = get_td_error(
        batch,
        primary_network,
        target_network,
        GAMMA
    )
    for i, idx in enumerate(idxs):
        memory.update(idx, errors[i])
    loss = primary_network.train_on_batch(
        batch.states, qt_for_train, sample_weight=is_weights)
    return loss


@click.command()
@click.option('--tc', type=int, default=20)
@click.option('--halfrange', type=int, default=2)
@click.option('--hidden_size', type=int, default=50)
@click.option('--action_space', type=int, default=2)
@click.option('--num_episodes', type=int, default=2000)
@click.option('--rc_min', type=int, default=50)
@click.option('--rc_max', type=int, default=100)
def main(tc, halfrange, hidden_size, action_space, num_episodes, rc_min, rc_max):
    """main command"""
    text = get_feed(1, INTERVAL, path=PATH)
    model = MODELS["cells"]
    model = set_extra_params(model, autosize_max_power=True)
    model = set_extra_params(model, action_space=action_space)
    env = getattr(energy_gym, "StepRewardVacancy")(text, MAX_POWER, tc, **model)

    primary_network = DQModel(hidden_size, action_space)
    target_network = DQModel(hidden_size, action_space)
    primary_network.compile(optimizer=keras.optimizers.Adam(), loss='mse')
    # make target_network = primary_network
    update_network(primary_network, target_network, coeff=1)

    memory = Memory(50000, env.observation_space.shape)
    memory.beta = MIN_BETA
    # uncomment next line if dont want to normalize the importance sampling weights
    #memory.normalize_is_weights = False

    eps = MAX_EPS
    suffix = f'{NOW}_{hidden_size}MLP'
    train_writer = tf.summary.create_file_writer(f'{STORE_PATH}/{suffix}')
    steps = 0
    for i in range(num_episodes):
        cnt = 1
        avg_loss = 0
        tot_reward = 0

        state = gen_random_model_and_reset(
            env,
            "synth",
            tc=tc,
            halfrange=halfrange,
            rc_min=rc_min,
            rc_max=rc_max
        )
        print("*****************************************************************")
        print(f'nombre des samples dans la mémoire : {memory.available_samples}')
        print(f'position du pointeur d\'écriture : {memory.curr_write_idx}')
        while True:
            action = choose_action(state, primary_network, eps, steps, action_space)
            next_state, reward, done, _ = env.step(action)
            tot_reward += reward
            loss = -1
            # default priority is 1, used during pre-training
            priority = 1
            if steps > DELAY_TRAINING:
                loss = train(primary_network, memory, target_network)
                update_network(primary_network, target_network, coeff=TAU)
                if next_state is None:
                    next_state = np.zeros(state.shape)
                batch = Batch(
                    states=np.array([state]),
                    next_states=np.array([next_state]),
                    actions=np.array([action]),
                    rewards=np.array([reward]),
                    terminal=np.array([done])
                )
                _, error = get_td_error(
                    batch,
                    primary_network,
                    target_network,
                    GAMMA
                )
                priority = error[0]
            # store experience in memory
            memory.append((state, action, reward, done), priority)
            avg_loss += loss

            # linearly decay the eps and PER beta values
            eps = linear_decay(steps, MAX_EPS, MIN_EPS, DELAY_TRAINING, EPS_ITER)
            beta = linear_decay(steps, MIN_BETA, MAX_BETA, DELAY_TRAINING, BETA_ITER)
            memory.beta = beta

            steps += 1

            if done:
                if steps > DELAY_TRAINING:
                    avg_loss /= cnt
                    message = f'Episode: {i}, Reward: {reward:.2f}'
                    message = f'{message}, Total reward {tot_reward:.2f}'
                    message = f'{message}, avg loss: {avg_loss:.5f}, eps: {eps:.3f}'
                    print(message)
                    show_episode_stats(env)
                    add_scalars_to_tensorboard(train_writer, i, reward, avg_loss, env)
                    with train_writer.as_default():
                        tf.summary.scalar('beta', memory.beta, step=i)
                        tf.summary.scalar('eps', eps, step=i)
                else:
                    print(f'Pre-training...Episode: {i}')
                break

            state = next_state
            cnt += 1

    primary_network.save(f'{STORE_PATH}_{suffix}')


if __name__ == "__main__":
    main()
