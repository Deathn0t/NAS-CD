import collections
import tensorflow as tf
from deephyper.search.nas.model.space import KSearchSpace
from deephyper.search.nas.model.space.node import ConstantNode, VariableNode
from deephyper.search.nas.model.space.op.basic import Tensor
from deephyper.search.nas.model.space.op.connect import Connect
from deephyper.search.nas.model.space.op.merge import Concatenate
from deephyper.search.nas.model.space.op.op1d import Dense, Identity


def create_search_space(input_shape=(2,), output_shape=(5,), *args, **kwargs):

    ss = KSearchSpace(input_shape, output_shape)
    x = ss.input_nodes[0]

    hid_2 = ConstantNode(op=Dense(8, "relu"), name="hid_2")
    out_2 = ConstantNode(op=Dense(1), name="out_2")
    ss.connect(hid_2, out_2)

    hid_3 = ConstantNode(op=Dense(8, "relu"), name="hid_3")
    out_3 = ConstantNode(op=Dense(1), name="out_3")
    ss.connect(hid_3, out_3)

    hid_4 = ConstantNode(op=Dense(8, "relu"), name="hid_4")
    out_4 = ConstantNode(op=Dense(1), name="out_4")
    ss.connect(hid_4, out_4)

    hid_5 = ConstantNode(op=Dense(8, "relu"), name="hid_5")
    out_5 = ConstantNode(op=Dense(1), name="out_5")
    ss.connect(hid_5, out_5)

    hid_6 = ConstantNode(op=Dense(8, "relu"), name="hid_6")
    out_6 = ConstantNode(op=Dense(1), name="out_6")
    ss.connect(hid_6, out_6)

    # L1 DEPENDENT ON DATA OVERALL
    in_2 = VariableNode(name="in_2")
    in_2.add_op(Concatenate(ss, [x]))
    in_2.add_op(Concatenate(ss, [x, out_3]))
    in_2.add_op(Concatenate(ss, [x, out_4]))
    in_2.add_op(Concatenate(ss, [x, out_5]))
    in_2.add_op(Concatenate(ss, [x, out_6]))

    ss.connect(in_2, hid_2)

    # L2 DEPENDANT ON DATA, L1 AND L3
    in_3 = VariableNode(name="in_3")
    in_3.add_op(Concatenate(ss, [x, out_2]))
    in_3.add_op(Concatenate(ss, [x, out_4]))
    
    ss.connect(in_3, hid_3)

    # L3 DEPENDANT ON DATA, L1 L2 AND WIDTH
    in_4 = VariableNode(name="in_4")
    in_4.add_op(Concatenate(ss, [x]))
    in_4.add_op(Concatenate(ss, [x, out_2]))
    in_4.add_op(Concatenate(ss, [x, out_3]))
    in_4.add_op(Concatenate(ss, [x, out_5]))
    in_4.add_op(Concatenate(ss, [x, out_6]))
    in_4.add_op(Concatenate(ss, [x, out_2,out_3]))
    in_4.add_op(Concatenate(ss, [x, out_2,out_5]))
    in_4.add_op(Concatenate(ss, [x, out_2,out_6]))
    in_4.add_op(Concatenate(ss, [x, out_3,out_5]))
    in_4.add_op(Concatenate(ss, [x, out_3,out_6]))
    in_4.add_op(Concatenate(ss, [x, out_5,out_6]))
    in_4.add_op(Concatenate(ss, [x, out_3, out_5,out_6]))
    in_4.add_op(Concatenate(ss, [x, out_2, out_5,out_6]))
    in_4.add_op(Concatenate(ss, [x, out_2, out_3,out_6]))
    in_4.add_op(Concatenate(ss, [x, out_2, out_3,out_5]))
    in_4.add_op(Concatenate(ss, [x, out_2, out_3,out_5, out_6]))

    ss.connect(in_4, hid_4)

    # HEIGHT DEPENDANT ON ALL MEASURES COMBINED AND DATA
    in_5 = VariableNode(name="in_5")
    in_5.add_op(Concatenate(ss, [x]))
    in_5.add_op(Concatenate(ss, [x, out_2]))
    in_5.add_op(Concatenate(ss, [x, out_3]))
    in_5.add_op(Concatenate(ss, [x, out_4]))
    in_5.add_op(Concatenate(ss, [x, out_6]))
    in_5.add_op(Concatenate(ss, [x, out_2,out_3]))
    in_5.add_op(Concatenate(ss, [x, out_2,out_4]))
    in_5.add_op(Concatenate(ss, [x, out_2,out_6]))
    in_5.add_op(Concatenate(ss, [x, out_3,out_4]))
    in_5.add_op(Concatenate(ss, [x, out_3,out_6]))
    in_5.add_op(Concatenate(ss, [x, out_4,out_6]))
    in_5.add_op(Concatenate(ss, [x, out_3, out_4,out_6]))
    in_5.add_op(Concatenate(ss, [x, out_2, out_4,out_6]))
    in_5.add_op(Concatenate(ss, [x, out_2, out_3,out_6]))
    in_5.add_op(Concatenate(ss, [x, out_2, out_3,out_4]))
    in_5.add_op(Concatenate(ss, [x, out_2, out_3,out_4, out_6]))
    ss.connect(in_5, hid_5)

# WIDTH DEPENDANT ON DATA AND CROSS SECTION 
    in_6 = VariableNode(name="in_6")
    in_6.add_op(Concatenate(ss, [x]))
    in_6.add_op(Concatenate(ss, [x, out_2]))
    in_6.add_op(Concatenate(ss, [x, out_3]))
    in_6.add_op(Concatenate(ss, [x, out_4]))
    in_6.add_op(Concatenate(ss, [x, out_5]))
    ss.connect(in_6, hid_6)

    out = ConstantNode(name="OUT")
    out.set_op(Concatenate(ss, stacked_nodes=[out_2,out_3,out_4,out_5,out_6]))


    return ss


def test_create_search_space():
    """Generate a random neural network from the search_space definition.
    """
    from random import random
    from tensorflow.keras.utils import plot_model
    import tensorflow as tf

    search_space = create_search_space()
    # ops = [random() for _ in range(search_space.num_nodes)]
    # ops = [0. for _ in range(search_space.num_nodes)]
    ops = [0, 13, 7, 0, 0]
    #ops = [1, 0, 0]

    print(f"This search_space needs {len(ops)} choices to generate a neural network.")
    print(f"Search space size is: {search_space.size}")

    search_space.set_ops(ops)
    search_space.draw_graphviz("sampled_neural_network.dot")

    model = search_space.create_model()
    model.summary()

    plot_model(model, to_file="sampled_neural_network.png", show_shapes=True)
    print("The sampled_neural_network.png file has been generated.")


if __name__ == "__main__":
    test_create_search_space()
