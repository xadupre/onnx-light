.. _l-cli-optimize:

optimize
========

Optimizes one ONNX model through
:class:`~onnx_light.onnx_core.optimization.GraphGraph`. Both entry points accept
the same arguments:

.. code-block:: bash

    onnx-light optimize models/model.onnx --output models/model.opt.onnx
    python -m onnx_light optimize model.onnx -o model.opt.onnx --verbose

The output is required and must be in the **same directory** as the input.
Paths are interpreted relative to the current working directory. A destination
in another directory is rejected, including a directory reached through a
symbolic link. Existing files and symbolic links are never overwritten, so
the input and weight files cannot accidentally be replaced.

Large external tensors remain external references with their original
locations, offsets and lengths. Existing weight files are neither copied nor
rewritten. External tensors smaller than 128 bytes are loaded inline for shape
inference; inline weights are retained. Constants introduced by optimization
are saved inline in the output model. This command optimizes a model, not a
pipeline.

Pattern selection
-----------------

Omitting ``--patterns`` uses the optimizer's default patterns. To select a
registered group or the union of several groups, separate their names with
commas:

.. code-block:: bash

    onnx-light optimize model.onnx -o model.opt.onnx --patterns L1
    onnx-light optimize model.onnx -o model.opt.onnx --patterns L1,L2

The registered groups are exposed by
:func:`~onnx_light.onnx_core.optimization.standard_pattern_sets`.
``Layer`` selects the opt-in ``RepeatedLayer`` pattern; ``all`` includes every
registered pattern, including opt-in patterns, and is not the default selection.

Explicit pattern names use the same comma-separated syntax. Groups and
individual names may also be combined:

.. code-block:: bash

    onnx-light optimize model.onnx -o model.opt.onnx --patterns NotNot,CastCast
    onnx-light optimize model.onnx -o model.opt.onnx --patterns L1,TransposeTranspose

Names are case-sensitive and exact, not regular expressions. Surrounding
whitespace is ignored, duplicates are removed, and unknown names or empty list
items are rejected before loading the model. The optimizer's ordinary cleanup
and constant-folding passes still run with an explicit selection.
