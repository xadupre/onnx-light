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
the input and weight files cannot accidentally be replaced. Recorded external
weight locations are also protected when the files are missing, including
references in nested graphs and local functions. The output is binary ONNX;
``.textproto``, ``.pipeline`` and ``.pipe-onnx`` output suffixes are rejected.
Serialization uses a temporary directory next to the destination and publishes
the completed model through an exclusive hard link. Failed writes leave no
partial destination and can be retried; publishing requires hard-link support
on the destination filesystem.

Large external tensors remain external references with their original
locations, offsets and lengths. Existing weight files are neither copied nor
rewritten. Numeric external tensors smaller than 128 bytes are loaded inline
for shape inference, including textproto inputs and tensors without explicit
length metadata. Reads are bounded to the tensor's size, not the remaining
weight file; inline weights are retained. Folding and Constant-to-initializer
conversion skip external payloads that were not loaded. Constants introduced
by optimization are saved inline in the output model. This command optimizes
a model, not a pipeline. Models containing ``training_info`` are rejected
before parsing because their training graphs cannot currently be preserved.

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

Names include patterns registered through ``register_pattern()`` in the calling
Python process. They are case-sensitive and exact, not regular expressions.
Surrounding whitespace is ignored, duplicates are removed, and unknown names or
empty list items are rejected before loading the model. The optimizer's ordinary
cleanup and constant-folding passes still run with an explicit selection.
Group members are validated too, including registrations that were removed
after a group was created.

Optimization report
-------------------

Use ``--report PATH`` to export the optimizer's report as UTF-8 text:

.. code-block:: bash

    onnx-light optimize model.onnx -o model.opt.onnx --report optimization.txt

The report includes iteration and rewrite counts, phase timings in nanoseconds,
per-pattern matching statistics and nested-subgraph statistics. Its parent
directory must exist but need not be the model directory. Existing files and
symbolic links, the output model path and recorded external weight locations
are protected. Like the model, the report is published through an exclusive
hard link so failed writes leave no partial report. The model is saved first;
if report publication fails, the completed model remains available.

Text-defined function patterns
------------------------------

``--function-pattern TEMPLATE REPLACEMENT`` accepts two **quoted ONNX function
texts**, not filenames or Python code. Each is parsed with
:func:`~onnx_light.onnx.parser.parse_function` and used to construct a
:class:`~onnx_light.onnx_core.optimization.FunctionPattern`. For example:

.. code-block:: bash

    onnx-light optimize model.onnx -o model.opt.onnx \
        --function-pattern \
        '<opset_import: ["": 18], domain: "patterns">
         DoubleNeg (x) => (y) {
             middle = Neg(x)
             y = Neg(middle)
         }' \
        '<opset_import: ["": 18], domain: "patterns">
         KeepInput (x) => (y) {
             y = Identity(x)
         }'

Repeat ``--function-pattern`` to add several custom rewrites. They are added
to the default patterns, or to the explicit ``--patterns`` selection, with
priority 0. Registrations are local to this optimization; the process-global
pattern registry is not modified. Invalid function texts and incompatible
template/replacement interfaces are rejected before loading the model.
Function inputs and outputs correspond by position, and operator domains,
attributes and opsets follow the usual ``FunctionPattern`` matching rules.
External tensor references in these texts are relative to the input model's
directory and receive the same bounded small-tensor loading and output-location
protection as model tensors.

Replacements are inlined by default. Add ``--function-pattern-as-function``
to emit model-local replacement function calls for all supplied pairs.
Passing the same text as both template and replacement in this mode
encapsulates the matched body without changing its computation. This flag
requires at least one ``--function-pattern``.
