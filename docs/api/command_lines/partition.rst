partition
=========

Extracts local functions while preserving inline weights and leaving large
external weight payloads unloaded. Chooses
repeated-layer detection or transfer-aware partitioning, saves a rewritten ONNX
model and prints a function summary.

.. code-block:: bash

    python -m onnx_light partition layer model.onnx --output model.layers.onnx
    python -m onnx_light partition mincut model.onnx --output model.partition.onnx

See :ref:`l-cli-partition` for algorithm descriptions, weight preservation
and output placement.

Both methods print their selection policy and two totals per call: inputs and
outputs, each with symbolic logical bytes and transfer bytes. Constants contribute
logical bytes but no transfer bytes, except for overridable initializers. The
report omits individual tensors, also with ``--verbose``. Unavailable volumes are
explicit unknowns. These post-internalization formulas are descriptive, not compared by
the selectors.

Command-line reference
----------------------

.. runpython::

    from onnx_light.__main__ import _build_parser

    parser = _build_parser()
    print(parser._subparsers._group_actions[0].choices["partition"].format_help())

Repeated layers
---------------

``partition layer`` retains the former ``layers`` arguments:
``model``, ``--output``/``-o``, ``--tiny-tensor-threshold`` and ``--verbose``/``-v``.

.. runpython::

    from onnx_light.__main__ import _build_parser

    partition = _build_parser()._subparsers._group_actions[0].choices["partition"]
    print(partition._subparsers._group_actions[0].choices["layer"].format_help())

Transfer-aware groups
---------------------

``partition mincut`` accepts the same arguments plus an optional
``--max-boundary-bytes`` cap (no cap by default), ``--min-group-nodes`` (default ``2``) and
``--max-group-nodes`` (default ``128``). The minimum must be at least two and
must not exceed the maximum. Symbolic boundaries are supported by default:
the algorithm minimizes unknown transfers first, then known bytes.
An explicit byte cap requires fully known boundary sizes.
Each partition is also limited to ``floor(model_weights / 20)`` distinct constant
inputs before internalization. This 5% budget counts weights, not their byte
volumes. Over-budget operators remain outside extracted functions.

.. runpython::

    from onnx_light.__main__ import _build_parser

    partition = _build_parser()._subparsers._group_actions[0].choices["partition"]
    print(partition._subparsers._group_actions[0].choices["mincut"].format_help())
