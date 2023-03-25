"""graph a tensorflow model
https://stackoverflow.com/questions/56690089/how-to-graph-tf-keras-model-in-tensorflow-2-0
"""
import tensorflow as tf
from tensorflow import keras
from dueling import DQModel


@tf.function
def trace(x):
    """execution trace function"""
    return primary_network(x)


def graph(primary_network):
    """generate the graph model"""
    logdir = "TensorBoard/graph"
    train_writer = tf.summary.create_file_writer(logdir)
    tf.summary.trace_on(graph=True, profiler=False)
    trace(tf.zeros((1,4)))
    with train_writer.as_default():
        tf.summary.trace_export(name="dueling_graph", step=0)


if __name__ == "__main__":
    primary_network = DQModel(50, 2)
    primary_network.compile(optimizer=keras.optimizers.Adam(), loss='mse')
    graph(primary_network)
