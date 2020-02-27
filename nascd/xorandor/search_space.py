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

    # z = ConstantNode(op=Dense(4, activation="relu"), name="Z")
    # ss.connect(x, z)
    # x = z

    out_xor = ConstantNode(op=Dense(1, activation="sigmoid"), name="out_XOR")

    out_and = ConstantNode(op=Dense(1, activation="sigmoid"), name="out_AND")

    out_or = ConstantNode(op=Dense(1, activation="sigmoid"), name="out_OR")

    in_xor = VariableNode(name="in_XOR")
    in_xor.add_op(Concatenate(ss, [x]))
    in_xor.add_op(Concatenate(ss, [x, out_and]))
    in_xor.add_op(Concatenate(ss, [x, out_or]))
    in_xor.add_op(Concatenate(ss, [x, out_and, out_or]))
    ss.connect(in_xor, out_xor)

    in_and = VariableNode(name="in_AND")
    in_and.add_op(Concatenate(ss, [x]))
    in_and.add_op(Concatenate(ss, [x, out_xor]))
    in_and.add_op(Concatenate(ss, [x, out_or]))
    in_and.add_op(Concatenate(ss, [x, out_xor, out_or]))
    ss.connect(in_and, out_and)

    in_or = VariableNode(name="in_OR")
    in_or.add_op(Concatenate(ss, [x]))
    in_or.add_op(Concatenate(ss, [x, out_xor]))
    in_or.add_op(Concatenate(ss, [x, out_and]))
    in_or.add_op(Concatenate(ss, [x, out_xor, out_and]))
    ss.connect(in_or, out_or)

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
    # ops = [random() for _ in range(search_space.num_nodes)]
    ops = [2, 0, 0]

    print(f"This search_space needs {len(ops)} choices to generate a neural network.")

    search_space.set_ops(ops)
    search_space.draw_graphviz("sampled_neural_network.dot")

    model = search_space.create_model()
    model.summary()

    plot_model(model, to_file="sampled_neural_network.png", show_shapes=True)
    print("The sampled_neural_network.png file has been generated.")


if __name__ == "__main__":
    test_create_search_space()
