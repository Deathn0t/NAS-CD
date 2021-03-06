import collections

import tensorflow as tf

from deephyper.search.nas.model.space import KSearchSpace
from deephyper.search.nas.model.space.node import ConstantNode, VariableNode
from deephyper.search.nas.model.space.op.basic import Tensor
from deephyper.search.nas.model.space.op.connect import Connect
from deephyper.search.nas.model.space.op.merge import Concatenate
from deephyper.search.nas.model.space.op.op1d import Dense, Identity


def create_search_space(input_shape=(2,), output_shape=(3,), *args, **kwargs):

    ss = KSearchSpace(input_shape, output_shape)
    x = ss.input_nodes[0]

    out_xor = ConstantNode(op=Dense(1), name="XOR")
    ss.connect(x, out_xor)

    out_and = ConstantNode(op=Dense(1), name="AND")
    ss.connect(x, out_and)

    out_or = ConstantNode(op=Dense(1), name="OR")
    ss.connect(x, out_or)

    out = ConstantNode(name="OUT")
    out.set_op(Concatenate(ss, stacked_nodes=[out_xor, out_and, out_or]))

    return ss


def test_create_search_space():
    """Generate a random neural network from the search_space definition.
    """
    from random import random
    from tensorflow.keras.utils import plot_model
    import tensorflow as tf

    search_space = create_search_space()
    ops = [random() for _ in range(search_space.num_nodes)]

    print(f"This search_space needs {len(ops)} choices to generate a neural network.")

    search_space.set_ops(ops)
    search_space.draw_graphviz("sampled_neural_network.dot")

    model = search_space.create_model()
    model.summary()

    plot_model(model, to_file="sampled_neural_network.png", show_shapes=True)
    print("The sampled_neural_network.png file has been generated.")


if __name__ == "__main__":
    test_create_search_space()
