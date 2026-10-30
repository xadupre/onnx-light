partition
=========

Extracts local functions without loading large raw weight payloads. Chooses
repeated-layer detection or transfer-aware partitioning, saves a rewritten ONNX
model and prints a function summary.

.. code-block:: bash

    python -m onnx_light partition layer model.onnx --output model.layers.onnx
    python -m onnx_light partition mincut model.onnx --output model.partition.onnx

See :ref:`l-cli-partition` for algorithm descriptions, structural export details
and output placement.

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

``partition mincut`` accepts the same arguments plus ``--max-boundary-bytes``
(default ``1024``), ``--min-group-nodes`` (default ``2``) and
``--max-group-nodes`` (default ``128``). The minimum must be at least two and
must not exceed the maximum.

.. runpython::

    from onnx_light.__main__ import _build_parser

    partition = _build_parser()._subparsers._group_actions[0].choices["partition"]
    print(partition._subparsers._group_actions[0].choices["mincut"].format_help())
