.. _l-howto-command-line:

:html_theme.sidebar_secondary.remove:

Command-line interface
======================

``onnx-light`` ships a small command-line interface that can be invoked as a
Python module::

    python -m onnx_light <subcommand> [options]

.. _l-cli-fillshape:

fillshape
---------

Fills a model's ``graph.value_info`` and ``graph.output`` with the shapes
inferred by the ``onnx_optim`` shape-inference engine, then writes the result
back to disk.

.. code-block:: bash

    python -m onnx_light fillshape model.onnx

Synopsis::

    python -m onnx_light fillshape MODEL [--output OUTPUT] [--keep]
                                         [--inplace-info] [--shape-tag] [--show]
                                         [--verbose [LEVEL]]

Positional argument
^^^^^^^^^^^^^^^^^^^

``MODEL``
    Path to the input ``.onnx`` model file.

Options
^^^^^^^

``--output OUTPUT`` / ``-o OUTPUT``
    Write the result to *OUTPUT* instead of overwriting *MODEL* in place.

    When the model stores weights in a separate file (external data), the
    output ``.onnx`` file is always placed **next to the original model**
    regardless of the directory given to ``--output``, so that the
    relative weight-file paths encoded in the proto remain valid.
    The weight file is **not** written again.

``--keep``
    Seed the inference context from any shapes already present in the
    model's ``graph.value_info`` and ``graph.output``
    (``prefill_with_value_info_output=True``).  Existing non-conflicting
    shapes are kept as anchors.

``--inplace-info``
    After shape inference, compute in-place buffer-reuse opportunities
    and record them in each eligible node's ``metadata_props`` under the
    key ``onnx_light.inplace_reuse``.

``--shape-tag``
    After shape inference, infer semantic ``shape``/``axes``/``weight``
    tags for every value and node in the graph and record them in
    ``metadata_props`` (keys ``onnx_light.value_tags`` and
    ``onnx_light.node_tag``).

``--show``
    Print the model with inferred shapes to stdout using
    :func:`~onnx_light.tools.pretty_print.pretty_onnx`; do **not** save
    the model.

``--verbose [LEVEL]``
    Prints shape-inference progress information.

    When passed without a level (``--verbose``), level ``1`` is used and
    a short summary is printed.
    Level ``2`` (``--verbose 2``) also prints per-event details.

Examples
^^^^^^^^

Fill shapes and overwrite the file in place:

.. code-block:: bash

    python -m onnx_light fillshape model.onnx

Write the result to a separate file:

.. code-block:: bash

    python -m onnx_light fillshape model.onnx -o model_with_shapes.onnx

Preserve existing symbolic dimensions as anchors:

.. code-block:: bash

    python -m onnx_light fillshape model.onnx --keep

Annotate nodes with in-place buffer-reuse information:

.. code-block:: bash

    python -m onnx_light fillshape model.onnx --inplace-info

Annotate values and nodes with semantic shape/axes/weight tags:

.. code-block:: bash

    python -m onnx_light fillshape model.onnx --shape-tag

Print inferred shapes without saving:

.. code-block:: bash

    python -m onnx_light fillshape model.onnx --show

Print shape-inference events:

.. code-block:: bash

    python -m onnx_light fillshape model.onnx --verbose
    python -m onnx_light fillshape model.onnx --verbose 2

External-data models
^^^^^^^^^^^^^^^^^^^^

When a model stores its weight tensors in a separate ``.data`` file,
``fillshape`` handles it transparently:

- The model is loaded **without** reading the weight bytes (they are not
  needed for shape inference).
- When ``--output`` is used, the output ``.onnx`` file is placed in the
  **same directory as the input model** so that the relative paths to the
  weight file remain correct.  No new weight file is created.

.. code-block:: bash

    # model.onnx references model.onnx.data for its weights
    python -m onnx_light fillshape model.onnx -o model_filled.onnx
    # model_filled.onnx is written next to model.onnx (not in the cwd)
    # model.onnx.data is untouched

See also
^^^^^^^^

* :func:`~onnx_light.onnx_optim.shape_inference.infer_shapes_model` — the
  Python function used under the hood.
* :ref:`l-howto-use-custom-shape-inference` — how to plug in a callback for
  custom operators before running shape inference.
* :ref:`l-how-to` — other onnx-light how-to recipes.
