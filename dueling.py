"""dueling Q"""
import random
import datetime as dt
import click
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.initializers import he_normal  # pylint: disable=E0401
from tensorflow.keras.layers import Dense  # pylint: disable=E0401

import conf
from conf import MODELS
from conf import PATH, MAX_POWER
import energy_gym
from energy_gym import get_feed, set_extra_params
from standalone_d_dqn import Memory, choose_action

# pylint: disable=no-value-for-parameter

GAME = "Heat"
DIR = "TensorBoard/DuelingQ"
STORE_PATH = f'{DIR}/{GAME}'
MAX_EPS = 1
MIN_EPS = 0.01
EPS_MIN_ITER = 5000
DELAY_TRAINING = 300
GAMMA = 0.95
BATCH_SIZE = 50
TAU = 0.08

INTERVAL = 3600
NOW = dt.datetime.now().strftime('%d%m%Y%H%M')

def linear_decay(steps, max_eps, min_eps, delay, eps_min_iter):
    """linearly reduce the eps value"""
    eps = max_eps
    if steps > delay:
        eps = min_eps
        if steps < eps_min_iter:
            eps = max_eps - (max_eps - min_eps) * (steps - delay) / eps_min_iter
    return eps


def update_network(primary_network, target_network, coeff=TAU):
    """update target network parameters slowly from primary network"""
    for t_tv, p_tv in zip(target_network.trainable_variables,
                          primary_network.trainable_variables):
        t_tv.assign(t_tv * (1 - coeff) + p_tv * coeff)


class MeanSubstraction(keras.layers.Layer):
    """mean substraction layer"""
    def call(self, inputs):  # pylint: disable=W0221
        """layer's logic"""
        return inputs - tf.reduce_mean(inputs)


class DQModel(keras.Model):
    """dueling Q network"""
    def __init__(self, hidden_size: int, num_actions: int):
        super().__init__()
        args = {"activation": "relu", "kernel_initializer": he_normal()}
        self.dense1 = Dense(hidden_size, **args)
        self.dense2 = Dense(hidden_size, **args)
        self.adv_dense = Dense(hidden_size, **args)
        self.adv_out = Dense(num_actions, kernel_initializer=he_normal())
        self.v_dense = Dense(hidden_size, **args)
        self.v_out = Dense(1, kernel_initializer=he_normal())
        # formula 9 of the original article implementation
        self.normalized_as_9 = MeanSubstraction()
        self.combine = keras.layers.Add()

    def call(self, inputs):  # pylint: disable=W0221
        """model's logic"""
        x = self.dense1(inputs)
        x = self.dense2(x)
        advantage = self.adv_dense(x)
        advantage = self.adv_out(advantage)
        value = self.v_dense(x)
        value = self.v_out(value)
        norm_advantage = self.normalized_as_9(advantage)
        combined = self.combine([value, norm_advantage])
        return combined


def train(primary_network, memory, target_network):
    """double DQN training"""
    batch = memory.sample(BATCH_SIZE)
    states = np.array([val[0] for val in batch])
    actions = np.array([val[1] for val in batch])
    rewards = np.array([val[2] for val in batch])
    next_states = np.array([(np.zeros(states.shape[1])
                             if val[3] is None else val[3]) for val in batch])
    # predict Q(s,a) given the batch of states
    # prim_qt = q at t with primary network
    prim_qt = primary_network(states)
    # predict Q(s',a') from the evaluation network
    # prim_qtp1 = q at t plus 1 with primary network
    prim_qtp1 = primary_network(next_states)
    # copy the prim_qt tensor into the target_q tensor
    # we then will update one index corresponding to the max action
    target_q = prim_qt.numpy()
    updates = rewards
    # axis samples
    smp_axis = tuple(range(1, len(next_states.shape)))
    valid_idxs = np.array(next_states).sum(axis=smp_axis) != 0
    batch_idxs = np.arange(BATCH_SIZE)
    # extract the best action from the next state
    prim_action_tp1 = np.argmax(prim_qtp1.numpy(), axis=1)
    # get all the q values for the next state
    q_from_target = target_network(next_states)
    # add the discounted estimated reward from the selected action (prim_action_tp1)
    updates[valid_idxs] += GAMMA * q_from_target.numpy()[
        batch_idxs[valid_idxs],
        prim_action_tp1[valid_idxs]
    ]
    # update the q target to train towards
    target_q[batch_idxs, actions] = updates
    # run a training batch
    loss = primary_network.train_on_batch(states, target_q)
    return loss

@click.command()
@click.option('--tc', type=int, default=20)
@click.option('--halfrange', type=int, default=2)
@click.option('--hidden_size', type=int, default=50)
@click.option('--action_space', type=int, default=2)
@click.option('--num_episodes', type=int, default=3000)
@click.option('--rc_min', type=int, default=50)
@click.option('--rc_max', type=int, default=100)
def main(tc, halfrange, hidden_size, action_space, num_episodes, rc_min, rc_max):
    """main command"""
    text = get_feed(1, INTERVAL, path=PATH)
    modelkey = "synth"
    defmodel = conf.generate(bank_name=modelkey)
    model = MODELS.get(modelkey, defmodel)
    model = set_extra_params(model, action_space=action_space)
    model = set_extra_params(model, autosize_max_power=True)
    env = getattr(energy_gym, "StepRewardVacancy")(text, MAX_POWER, tc, **model)

    num_actions = env.action_space.n
    eps = MAX_EPS

    primary_network = DQModel(hidden_size, num_actions)
    target_network = DQModel(hidden_size, num_actions)
    primary_network.compile(optimizer=keras.optimizers.Adam(), loss='mse')
    # make target_network = primary_network
    update_network(primary_network, target_network, coeff=1)

    memory = Memory(50000)
    suffix = f'DuelingQ_{NOW}_{hidden_size}MLP'
    tw_path = f'{STORE_PATH}/{suffix}'
    train_writer = tf.summary.create_file_writer(tw_path)
    steps = 0
    for i in range(num_episodes):
        cnt = 1
        avg_loss = 0
        # random model generation
        newmodel = conf.generate(bank_name=modelkey, rc_min=rc_min, rc_max=rc_max)
        env.update_model(newmodel)
        # model visualisation
        conf.output_model(env.model)
        max_power = round(env.max_power * 1e-3)
        print(f'max power : {max_power} kW')
        # random tc for the episode
        tc_episode = tc + random.randint(-halfrange, halfrange)
        # reset the environnement
        state = env.reset(tc_episode=tc_episode)

        while True:
            action = choose_action(state, primary_network, eps, num_actions)
            next_state, reward, done, _ = env.step(action)
            if done:
                next_state = None
            # store in memory
            memory.add_sample((state, action, reward, next_state))

            if steps > DELAY_TRAINING:
                loss = train(primary_network, memory, target_network)
                update_network(primary_network, target_network)
            else:
                loss = -1
            avg_loss += loss

            eps = linear_decay(steps, MAX_EPS, MIN_EPS, DELAY_TRAINING, EPS_MIN_ITER)
            steps += 1

            if done:
                if steps > DELAY_TRAINING:
                    avg_loss /= cnt
                    message = f'Episode: {i}, Reward: {reward}'
                    message = f'{message}, avg loss: {avg_loss:.5f}, eps: {eps:.3f}'
                    print(message)
                    with train_writer.as_default():
                        tf.summary.scalar('reward', reward, step=i)
                        tf.summary.scalar('avg loss', avg_loss, step=i)
                else:
                    print(f"Pre-training...Episode: {i}")
                break

            state = next_state
            cnt += 1

    save = input("save ? Y=yes")
    if save == "Y":
        primary_network.save(f'{STORE_PATH}_{suffix}')


if __name__ == "__main__":
    main()
