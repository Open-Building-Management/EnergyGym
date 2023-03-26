"""shared classes and methods for reinforcement learning"""
import random
from dataclasses import dataclass
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.initializers import he_normal  # pylint: disable=E0401
from tensorflow.keras.layers import Dense  # pylint: disable=E0401


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


@dataclass
class Batch:
    """training batch :-)"""
    states : np.ndarray
    next_states : np.ndarray
    actions : np.ndarray
    rewards : np.ndarray
    terminal : np.ndarray


class Node:  #  pylint: disable=R0903
    """tree node"""
    def __init__(self, left, right, is_leaf: bool=False, idx=None):
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        self.value = sum(n.value for n in (left, right) if n is not None)
        self.parent = None
        self.idx = idx  # this value is only set for leaf nodes
        if left is not None:
            left.parent = self
        if right is not None:
            right.parent = self

    @classmethod
    def create_leaf(cls, value, idx):
        """create a bottom leaf at index idx"""
        leaf = cls(None, None, is_leaf=True, idx=idx)
        leaf.value = value
        return leaf


def create_tree(inputs: list):
    """create the tree from a list"""
    nodes = [Node.create_leaf(v, i) for i, v in enumerate(inputs)]
    leaf_nodes = nodes
    while len(nodes) > 1:
        inodes = iter(nodes)
        nodes = [Node(*pair) for pair in zip(inodes, inodes)]
    return nodes[0], leaf_nodes


def retrieve(value: float, node: Node):
    """leaf node recursive retrieval function given a value and
    the parent top node of the tree
    """
    if node.is_leaf:
        return node
    if node.left.value >= value:
        return retrieve(value, node.left)
    return retrieve(value - node.left.value, node.right)


def update(node: Node, new_value: float):
    """updates the value of a leaf node"""
    change = new_value - node.value
    node.value = new_value
    propagate_changes(change, node.parent)


def propagate_changes(change: float, node: Node):
    """propagates changes to the top parent node"""
    node.value += change
    if node.parent is not None:
        propagate_changes(change, node.parent)


class Memory:
    """experience replay memory"""
    def __init__(self, size: int, state_size: tuple):
        """initialize the memory"""
        self.size = size
        self.curr_write_idx = 0
        self.available_samples = 0
        self.state_size = state_size
        self.buffer = [(np.zeros(state_size,
                                 dtype=np.float32), 0.0, 0.0, 0.0) for i in range(self.size)]
        self.base_node, self.leaf_nodes = create_tree([0 for i in range(self.size)])
        self.frame_idx = 0
        self.action_idx = 1
        self.reward_idx = 2
        self.terminal_idx = 3
        self.beta = 0.4
        self.alpha = 0.6
        self.min_priority = 0.01
        # normalize the importance sampling weights
        self.normalize_is_weights = True

    def append(self, experience: tuple, delta: float):
        """append a new experience to the memory"""
        self.buffer[self.curr_write_idx] = experience
        self.update(self.curr_write_idx, delta)
        self.curr_write_idx += 1
        # reset the current writer position index if needed
        if self.curr_write_idx >= self.size:
            self.curr_write_idx = 0
        # max out available samples at the memory buffer size
        if self.available_samples < self.size:
            self.available_samples += 1

    def update(self, idx: int, delta: float):
        """updates value of leaf node number idx"""
        priority = np.power(delta + self.min_priority, self.alpha)
        update(self.leaf_nodes[idx], priority)

    def sample(self, idxs):
        """return the batch corresponding to some given indexes"""
        num_samples = len(idxs)
        state_size = self.state_size
        states = np.zeros((num_samples, *state_size), dtype=np.float32)
        next_states = np.zeros((num_samples, *state_size), dtype=np.float32)
        actions = np.zeros(num_samples, dtype=np.uint)
        rewards = np.zeros(num_samples, dtype=np.float32)
        terminal = np.zeros(num_samples, dtype=np.bool)
        for i, idx in enumerate(idxs):
            states[i] = self.buffer[idx][self.frame_idx]
            next_states[i] = self.buffer[idx + 1][self.frame_idx]
            actions[i] = self.buffer[idx][self.action_idx]
            rewards[i] = self.buffer[idx][self.reward_idx]
            terminal[i] = self.buffer[idx][self.terminal_idx]
        batch = Batch(
            states=states,
            next_states=next_states,
            actions=actions,
            rewards=rewards,
            terminal=terminal
        )
        return batch

    def sample_base(self, num_samples: int):
        """create a batch without using the tree"""
        sampled_idxs = []
        is_weights = np.ones(num_samples)
        sample_no = 0
        while sample_no < num_samples:
            sample_idx = random.randrange(0, self.available_samples - 1)
            # pour être sur d'avoir le "vrai" état suivant
            # après que la mémoire ait été remplie une première fois
            # et que le pointeur d'écriture soit repassé à 0
            if self.curr_write_idx == 0 or sample_idx != self.curr_write_idx - 1:
                sampled_idxs.append(sample_idx)
                sample_no += 1
        result = self.sample(sampled_idxs), sampled_idxs, is_weights
        return result

    def sample_tree(self, num_samples: int):
        """create a batch using the priorities as defined in the tree"""
        sampled_idxs = []
        # importance samplingg weights
        is_weights = []
        sample_no = 0
        #print(self.base_node.value)
        while sample_no < num_samples:
            sample_val = np.random.uniform(0, self.base_node.value)
            samp_node = retrieve(sample_val, self.base_node)
            if samp_node.idx < self.available_samples - 1:
                if self.curr_write_idx == 0 or samp_node.idx != self.curr_write_idx - 1:
                    sampled_idxs.append(samp_node.idx)
                    # probability of choosing the node
                    prob = samp_node.value / self.base_node.value
                    is_weights.append(self.available_samples * prob)
                    sample_no += 1
        # apply the beta factor
        is_weights = np.array(is_weights)
        is_weights = np.power(is_weights, -self.beta)
        if self.normalize_is_weights:
            # normalise so that the maximum is_weight < 1
            is_weights = is_weights / np.max(is_weights)
        result = self.sample(sampled_idxs), sampled_idxs, is_weights
        return result
