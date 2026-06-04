onnx\_light.tools.mermaid
========================

Convert an ONNX model or graph to a `Mermaid <https://mermaid.js.org/>`_
``flowchart`` diagram.  The output is a plain string that can be
embedded in Markdown or in a Sphinx page via the ``.. mermaid::``
directive.

Example:

.. runpython::
    :showcode:

    from onnx_light.onnx import helper, TensorProto
    from onnx_light.tools import to_mermaid

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])
    Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 3])
    graph = helper.make_graph(
        [
            helper.make_node("Add", ["X", "Y"], ["T"]),
            helper.make_node("Mul", ["T", "X"], ["Z"]),
        ],
        "g",
        [X, Y],
        [Z],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)]
    )
    print(to_mermaid(model))

Rendered as a Mermaid diagram:

.. runpython::
    :rst:

    from onnx_light.onnx import helper, TensorProto
    from onnx_light.tools import to_mermaid

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])
    Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 3])
    graph = helper.make_graph(
        [
            helper.make_node("Add", ["X", "Y"], ["T"]),
            helper.make_node("Mul", ["T", "X"], ["Z"]),
        ],
        "g",
        [X, Y],
        [Z],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)]
    )
    print(".. mermaid::")
    print()
    for line in to_mermaid(model).splitlines():
        print("    " + line)

API
+++

.. automodule:: onnx_light.tools.mermaid
    :members:
