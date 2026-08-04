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
inferred by the ``onnx_shapes`` shape-inference engine, then writes the result
back to disk.

.. code-block:: bash

    python -m onnx_light fillshape model.onnx

Synopsis::

    python -m onnx_light fillshape MODEL [--output OUTPUT] [--keep]
                                         [--inplace-info] [--release-info]
                                         [--shape-tag] [--show]
                                         [--token NAME=LOW:HIGH ...]
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

``--release-info``
    After shape inference, compute last-use release hints and record them
    in each eligible node's ``metadata_props`` under the key
    ``onnx_light.release_after``.

``--shape-tag``
    After shape inference, infer semantic ``shape``/``axes``/``weight``/``ambiguous``
    tags for every value and node in the graph and record them in
    ``metadata_props`` (per-value key ``onnx_light.value_tag`` on each
    ValueInfoProto/initializer, and per-node key ``onnx_light.node_tag``).

``--token NAME=LOW:HIGH``
    Bind a symbolic dimension token to an inclusive integer range before
    running shape inference.  May be specified multiple times.

    * ``--token seq=1:128`` — treats ``seq`` as having range ``[1, 128]``
      and uses ``1`` (the lower bound) for shape propagation.

    Symbolic dims not covered by ``--token`` remain symbolic.  Providing
    this option forces the use of the :class:`ShapesContext`-based path.

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

Bind symbolic dimensions to ranges for shape propagation:

.. code-block:: bash

    python -m onnx_light fillshape model.onnx --token batch=1:8 --token seq=1:128

Bind a symbolic dimension to a range (lower bound used for inference):

.. code-block:: bash

    python -m onnx_light fillshape model.onnx --token seq=1:512

Annotate nodes with in-place buffer-reuse information:

.. code-block:: bash

    python -m onnx_light fillshape model.onnx --inplace-info

Annotate nodes with release hints:

.. code-block:: bash

    python -m onnx_light fillshape model.onnx --release-info

Annotate values and nodes with semantic shape/axes/weight/ambiguous tags:

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

* :func:`~onnx_light.onnx_core.shape_inference.infer_shapes_model` — the
  Python function used under the hood.
* :ref:`l-howto-use-custom-shape-inference` — how to plug in a callback for
  custom operators before running shape inference.
* :ref:`l-how-to` — other onnx-light how-to recipes.

.. _l-cli-show:

show
----

Loads an ONNX model and renders it as plain text, a Mermaid flowchart, an
SVG image or Graphviz DOT source.  The result is written to stdout by default.

.. code-block:: bash

    python -m onnx_light show model.onnx

Synopsis::

    python -m onnx_light show MODEL [--format {pretty,mermaid,svg,dot}]
                                    [--output OUTPUT]
                                    [--shape-inference]
                                    [--no-shapes]
                                    [--include-attributes]
                                    [--include-inplace]
                                    [--include-node-tags]
                                    [--no-initializers]
                                    [--direction DIRECTION]
                                    [--graphviz GRAPHVIZ_FORMAT]

Positional argument
^^^^^^^^^^^^^^^^^^^

``MODEL``
    Path to the input ``.onnx`` model file.

Options
^^^^^^^

``--format {pretty,mermaid,svg,dot}`` / ``-f {pretty,mermaid,svg,dot}``
    Output format (default: ``pretty``).

    * ``pretty`` — compact text listing produced by
      :func:`~onnx_light.tools.pretty_print.pretty_onnx`.
    * ``mermaid`` — Mermaid flowchart source that can be embedded in Markdown
      or rendered with the Mermaid CLI.
    * ``svg`` — SVG image written to stdout (or to the file given by
      ``--output``).
    * ``dot`` — Graphviz DOT source produced by
      :func:`~onnx_light.tools.dot.to_dot`.

``--output OUTPUT`` / ``-o OUTPUT``
    Write the rendered output to *OUTPUT* instead of printing to stdout.

``--shape-inference``
    Run onnx-light shape inference on the model before rendering.

``--no-shapes``
    Suppress shape annotations in the rendered output
    (applies to ``mermaid``, ``svg`` and ``dot`` formats).

``--include-attributes``
    Include node attributes in the rendered output.

``--include-inplace``
    Show in-place buffer-reuse annotations (``onnx_light.inplace_reuse``
    metadata).

``--include-node-tags``
    Show semantic shape/axes/weight/ambiguous node-tag annotations
    (``onnx_light.node_tag`` metadata).  Only used by the ``pretty`` format.

``--no-initializers``
    Exclude initializer nodes from the rendered graph
    (applies to ``mermaid``, ``svg`` and ``dot`` formats).

``--direction DIRECTION``
    Flowchart direction for ``mermaid``, ``svg`` and ``dot`` formats.
    One of ``TB`` (default), ``LR``, ``TD`` or ``BT`` (Mermaid only).

``--graphviz GRAPHVIZ_FORMAT``
    Invoke the Graphviz ``dot`` executable on the generated DOT source and
    write the rendered image in *GRAPHVIZ_FORMAT* (e.g. ``png``, ``svg``,
    ``pdf``).  Only used when ``--format dot`` is given.  Requires Graphviz
    to be installed and ``dot`` to be available on ``PATH``.

Examples
^^^^^^^^

Print a compact text summary to stdout:

.. code-block:: bash

    python -m onnx_light show model.onnx

Run shape inference first, then show the model:

.. code-block:: bash

    python -m onnx_light show model.onnx --shape-inference

Render a Mermaid flowchart and save it to a file:

.. code-block:: bash

    python -m onnx_light show model.onnx --format mermaid -o model.mmd

Generate an SVG with a left-to-right layout:

.. code-block:: bash

    python -m onnx_light show model.onnx --format svg --direction LR -o model.svg

Show the model without initializer nodes and without shape annotations:

.. code-block:: bash

    python -m onnx_light show model.onnx --format mermaid --no-initializers --no-shapes

Dump the Graphviz DOT source to a file:

.. code-block:: bash

    python -m onnx_light show model.onnx --format dot -o model.dot

Render a PNG image via Graphviz (requires ``dot`` on PATH):

.. code-block:: bash

    python -m onnx_light show model.onnx --format dot --graphviz png -o model.png

See also
^^^^^^^^

* :func:`~onnx_light.tools.pretty_print.pretty_onnx` — the Python function
  used by the ``pretty`` format.
* :func:`~onnx_light.tools.dot.to_dot` — the Python function used by the
  ``dot`` format.
* :ref:`l-how-to` — other onnx-light how-to recipes.

.. _l-cli-run:

run
---

Generates random inputs for every graph input using the onnx-light
deterministic pseudo-random generators and runs the model through the
onnx-light runtime.

.. code-block:: bash

    python -m onnx_light run model.onnx

Synopsis::

    python -m onnx_light run MODEL [--dim NAME=VALUE ...]
                                   [--seed SEED]
                                   [--verbose [LEVEL]]
                                   [--dump PATH]

Positional argument
^^^^^^^^^^^^^^^^^^^

``MODEL``
    Path to the input ``.onnx`` model file.

Options
^^^^^^^

``--dim NAME=VALUE``
    Override a symbolic (dynamic) dimension by name.
    May be given multiple times (e.g. ``--dim batch=4 --dim seq=16``).
    Dimensions that have no concrete value and are not covered by ``--dim``
    default to ``1``.

``--seed SEED``
    Integer seed for the random input generator (default: ``0``).

``--verbose [LEVEL]`` / ``-v [LEVEL]``
    Print run progress.  When passed without a level (``--verbose``),
    level ``1`` is used.

    * Level ``1`` prints loading progress, input shapes and output shapes.
    * Level ``2`` also prints per-node execution details.

``--dump PATH``
    Write all inputs and outputs to *PATH* as an ONNX model whose
    ``graph.initializer`` contains one ``TensorProto`` per tensor.
    Non-array outputs (e.g. sequences) are skipped.

Examples
^^^^^^^^

Run a model with random inputs:

.. code-block:: bash

    python -m onnx_light run model.onnx

Set a specific batch size and sequence length for dynamic models:

.. code-block:: bash

    python -m onnx_light run model.onnx --dim batch=4 --dim seq=128

Use a fixed random seed for reproducible results:

.. code-block:: bash

    python -m onnx_light run model.onnx --seed 42

Print input/output shapes and execution progress:

.. code-block:: bash

    python -m onnx_light run model.onnx --verbose
    python -m onnx_light run model.onnx --verbose 2

Save the inputs and outputs to a file for later inspection:

.. code-block:: bash

    python -m onnx_light run model.onnx --dump io_tensors.onnx

See also
^^^^^^^^

* :class:`~onnx_light.onnx.reference.ReferenceEvaluator` — the Python
  evaluator used under the hood.
* :func:`~onnx_light.onnx.tools.make_random_input` — the public helper used
  to synthesize each input tensor.
* :ref:`l-how-to` — other onnx-light how-to recipes.
