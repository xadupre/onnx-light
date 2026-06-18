onnx_light.tools.svg
====================

.. automodule:: onnx_light.tools.svg
    :members:

Example
-------

The following snippet builds a small model and renders it with
:func:`onnx_light.tools.to_svg`. The resulting image is embedded below the
code so it can be inspected directly in the documentation.

.. runpython::
    :showcode:
    :rst:

    from onnx_light.onnx_lib import TensorProto
    from onnx_light.onnx.helper import (
        make_model,
        make_node,
        make_graph,
        make_tensor_value_info,
    )
    from onnx_light.tools import to_svg

    X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
    A = make_tensor_value_info("A", TensorProto.FLOAT, [None, None])
    B = make_tensor_value_info("B", TensorProto.FLOAT, [None, None])
    Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])

    node1 = make_node("MatMul", ["X", "A"], ["XA"])
    node2 = make_node("Add", ["XA", "B"], ["Y"])
    graph = make_graph([node1, node2], "example", [X, A, B], [Y])
    model = make_model(graph)

    svg = to_svg(model)

    # Embed the rendered SVG directly into the documentation page.
    print(".. raw:: html")
    print()
    for line in svg.splitlines():
        print(f"    {line}")
