onnx\_light.tools.mermaid
=========================

Convert an ONNX model or graph to a :epkg:`Mermaid`
``flowchart`` diagram.  The output is a plain string that can be
embedded in Markdown or in a Sphinx page via the ``.. mermaid::``
directive.

Example
+++++++

Building a tiny model and printing its Mermaid source — using
``sphinx_runpython``'s ``.. runpython:: :showcode:`` directive, which
executes the Python block at documentation build time and renders both
the code and its standard output:

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

Rendering the diagram
+++++++++++++++++++++

To actually display the diagram, ``sphinx_runpython`` ships a
``.. runmermaid::`` directive that wraps the Mermaid source into the
HTML output (which is then rendered client-side by :epkg:`mermaid.js`).
It accepts a ``:script:`` option: when
present, the directive body is executed as Python and its standard
output is used as the Mermaid source. This pairs naturally with
:func:`to_mermaid`, which already prints a complete ``flowchart``
definition.

The pattern is therefore::

    .. runmermaid::
        :script:

        # any Python code that ends with `print(to_mermaid(model))`
        from onnx_light.onnx import helper, TensorProto
        from onnx_light.tools import to_mermaid
        ...
        print(to_mermaid(model))

Applied to the model above:

.. runmermaid::
    :script:

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

If the Mermaid source is already available as a static string (for
instance copy-pasted from a previous run), the same ``.. runmermaid::``
directive can be used without ``:script:`` and the body is treated as
raw Mermaid::

    .. runmermaid::

        flowchart TB
            A --> B --> C

API
+++

.. automodule:: onnx_light.tools.mermaid
    :members:
