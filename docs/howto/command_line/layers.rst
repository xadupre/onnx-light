.. _l-cli-layers:

layers
======

Extracts repeated computations into model-local functions using only the
opt-in ``RepeatedLayer`` pattern:

.. code-block:: bash

    onnx-light layers model.onnx --output model.layers.onnx
    python -m onnx_light layers model.onnx -o model.layers.onnx --verbose

The output is a rewritten ONNX model, not one file per layer. Shared function
bodies appear in ``ModelProto.functions`` as ``ai.rt::Layer``, ``Layer_1``, etc.,
with distinct weights passed explicitly at each call. The command prints each
function's node count, input/output counts, and number of static call sites,
including calls inside nested graphs and local functions. It also reports the
number of extraction rewrites. A model without repeated regions is saved
normally and reports zero layers. Detection finds repeated computations, which
need not correspond to complete transformer layers. See
:ref:`l-design-optimization` for the pattern contracts.

Identical computations share a function even when exporter namespaces,
diagnostics or inferred symbolic shape names differ. Different computations
retain distinct functions. The summary counts ``Layer`` definitions, not the
other local functions already present in the model.

Loading and weights
-------------------

The binary ONNX model is parsed without reading large ``raw_data`` payloads
or large external tensor payloads. Raw tensors strictly smaller than 128 bytes
are retained; tiny external tensors are loaded and inlined so shape inference
can use values such as a ``Reshape`` shape. Change this positive byte threshold
with ``--tiny-tensor-threshold BYTES``. Typed protobuf payload fields remain
part of the model metadata and are parsed normally.

Large external weights retain their original location, offset and length;
the command never rewrites their files. Large inline raw tensor payloads are
omitted from this structural export. To retain executable weights in the
rewritten model, externalize them before running the command.
Results with omitted inline weights are for structural inspection only.
Constant folding is disabled; only ``RepeatedLayer`` and the optimizer's normal
structural cleanup passes run.

``--output`` is required and must name a **new file**. Existing files, including
the input model and weight files, are never overwritten. External weight paths
are preserved without validation or rebasing. Place the output in the input
model's directory, or arrange the external files relative to its new location.
Those external files must remain available when loading the rewritten model's
weights. Missing or invalid tiny shape tensors raise an error during loading.

.. code-block:: bash

    python -m onnx_light layers model.onnx \
        --output model.layers.onnx \
        --tiny-tensor-threshold 256

For the complete argument reference, see
:doc:`../../api/command_lines/layers`.
