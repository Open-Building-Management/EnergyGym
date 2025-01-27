"""graph a tensorflow model
https://stackoverflow.com/questions/56690089/how-to-graph-tf-keras-model-in-tensorflow-2-0
"""
import tensorflow as tf
from shared_rl_tools import DQModel


@tf.function
def trace(x):
    """execution trace function"""
    return primary_network(x)


if __name__ == "__main__":
    primary_network = DQModel(50, 2)
    primary_network.compile(optimizer=tf.optimizers.Adam(), loss='mse')
    train_writer = tf.summary.create_file_writer("TensorBoard/graph")
    tf.summary.trace_on(graph=True, profiler=False)
    trace(tf.zeros((1,4)))
    with train_writer.as_default():
        tf.summary.trace_export(name="dueling_graph", step=0)
