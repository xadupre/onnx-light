.. _l-cli-partition:

partition
=========

Extracts model-local functions using one of two partitioning methods. Both
methods preserve inline tensor payloads and external weight references
and use the same loading, output and verbosity arguments. The former ``layers``
command is now ``partition layer``.

Repeated layers
---------------

Extracts repeated computations into model-local functions using only the
opt-in ``RepeatedLayer`` pattern:

.. code-block:: bash

    onnx-light partition layer model.onnx --output model.layers.onnx
    python -m onnx_light partition layer model.onnx -o model.layers.onnx --verbose

The output is a rewritten ONNX model, not one file per layer. Shared function
bodies appear in ``ModelProto.functions`` as ``ai.rt::Layer_MatMul_0``,
``ai.rt::Layer_Attention_0``, etc. The operator name comes from the first anchor
in source order; the index starts at zero and avoids existing names.
Distinct weights are passed explicitly at each call. The command prints each
function's node count, input/output counts, and number of static call sites,
including calls inside nested graphs and local functions. It also reports the
number of extraction rewrites. A model without repeated regions is saved
normally and reports zero layers. Detection finds repeated computations, which
need not correspond to complete transformer layers. See
:ref:`l-design-optimization` for the pattern contracts.

For both methods, available constants with strictly fewer than 32 elements are
embedded as ``Constant`` nodes when their dtype, shape and values are common to
all calls of the function. They are removed from its arguments. Other constants,
including overridable initializers and unloaded external data, remain arguments;
extraction does not load additional weights for specialization.

Both methods also turn ``f(x, Shape(x))`` into ``f(x)`` by computing ``Shape(x)``
inside the function. The source must already be an input at the corresponding
position in every call, with matching ``Shape`` attributes. This works with
dynamic dimensions and preserves outside uses of the original shape value.

Identical computations share a function even when exporter namespaces,
diagnostics or inferred symbolic shape names differ. Different computations
retain distinct functions. The summary counts ``Layer`` definitions, not the
other local functions already present in the model.

Detection starts with rarer attention/projection anchors and grows complete
repeated motifs of up to 256 nodes. For a hybrid model, one function can therefore
contain several consecutive blocks, for example three linear-attention blocks
followed by one standard-attention block. Shared constant weight conversions
may move before those calls without loading or folding the weights.

Formal inputs use the longest prefix shared by a strict majority of the
corresponding arguments in the matched calls: ``A1``, ``A2``, ``B0`` give ``A``.
If that name is already used, the input position is appended, for example ``A_2``.
Without a majority prefix, the name is ``input_2`` for input position 2.

Transfer-aware partitioning
---------------------------

``partition mincut`` applies the opt-in ``PartitionGroup`` pattern. It groups
topological intervals into ``ai.rt::PartitionGroup_<mainop>_<index>`` functions
while minimizing boundary transfers. Despite the command name, this is bounded
interval dynamic programming, not a general graph minimum-cut solver.

.. code-block:: bash

    python -m onnx_light partition mincut model.onnx \
        --output model.partition.onnx \
        --min-group-nodes 2 \
        --max-group-nodes 128

There is **no byte cap by default**. Symbolic or otherwise unknown tensor sizes
do not block partitioning: the algorithm minimizes the number of unknown boundary
transfers first, then known transfer bytes, and finally ungrouped nodes.
Unknown sizes are not replaced by zero bytes or invented dimension values.
Constants and non-overridable initializers cost no transfer bytes.

Each partition may consume at most **5% of the model's weights**, counted as
distinct constant inputs, not bytes or elements. The command counts declared
non-overridable dense, sparse and extended initializers and standard ``Constant``
outputs across model scopes before optimization. A shared weight counts once
per partition. The limit is rounded down and checked before small constants
are internalized, so embedding them cannot bypass it. The selection summary
prints the model weight count and the resulting per-partition limit.
Operators that cannot satisfy the limit remain outside extracted functions;
for fewer than 20 weights, weighted partitions are therefore not extracted.

For static boundaries, optionally pass ``--max-boundary-bytes BYTES`` to require
each group's combined input/output bytes to fit that cap. An explicit cap
excludes groups with unknown boundary sizes.
``--min-group-nodes`` and ``--max-group-nodes`` bound each group, defaulting to
2 and 128. All limits must fit unsigned 64-bit integers.

Both methods share the same naming implementation: ``mainop`` is the first
anchor's operator type in source order, or the first operator when a mincut group
has no attention/MatMul anchor. Indices start at zero and skip existing function,
graph-value and registered-operator names. For example, a group containing
``Reshape``, ``MatMul`` and ``Relu`` becomes ``PartitionGroup_MatMul_0``.

The summary lists ``PartitionGroup`` definitions and their call sites. A model
without eligible groups is still saved and reports zero partition functions.
Equivalent partitions reuse a single function definition, using the same
body and interface comparison as repeated layers. Different attributes or
embedded constants retain distinct functions; external weights remain call arguments.
Each candidate definition is normalized once per extraction and indexed by a
fingerprint; matching still compares bodies exactly, including hash collisions.

Explaining interfaces
--------------------

Both ``partition layer`` and ``partition mincut`` print their selection policy
and report two totals per call, one for all inputs and one for all outputs,
after constant and ``Shape`` internalization. Logical byte volumes retain symbolic
dimensions, for example ``4*N`` for a ``FLOAT[N]`` tensor. Transfer volumes exclude constants and
non-overridable initializers, while their logical volumes remain visible.
Unavailable sizes appear explicitly as ``unknown[N]``, where ``N`` is their count.

The report does not list individual tensors, including with ``--verbose`` or
``-v 1``. Reporting uses metadata only and does not read additional weight payloads.
Missing shapes and unsupported types remain unknown rather than
receiving invented values. These formulas explain the resulting interfaces;
neither selector compares symbolic formulas as a new optimization objective.

Loading and weights
-------------------

All inline tensor payloads, including ``raw_data`` and typed protobuf fields,
are loaded and preserved regardless of size. External tensors strictly smaller
than 128 bytes are loaded and inlined so shape inference can use values such as
a ``Reshape`` shape. Change this positive byte threshold with
``--tiny-tensor-threshold BYTES``; it only controls external tensor loading.

Large external weights retain their original location, offset and length;
their payloads are not loaded during extraction. The rewritten model retains its
weights and can be verified and executed while its external files remain
available. For models with large inline weights, externalize them before
partitioning to reduce memory usage.
Constant folding is disabled; only the selected pattern and the optimizer's
normal structural cleanup passes run.

``--output`` is required. An existing output file is overwritten; ``--verbose``
announces the overwrite. Using the input model path rewrites it in place.
Choose an output path distinct from the external weight files. External weight
paths are preserved without validation or rebasing. Place the output in the input
model's directory, or arrange the external files relative to its new location.
Those external files must remain available when loading the rewritten model's
weights. Missing or invalid tiny shape tensors raise an error during loading.

.. code-block:: bash

    python -m onnx_light partition layer model.onnx \
        --output model.layers.onnx \
        --tiny-tensor-threshold 256

For the complete argument reference, see
:doc:`../../api/command_lines/partition`.
