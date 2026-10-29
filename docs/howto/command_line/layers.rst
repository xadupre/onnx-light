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

Loading and weights
-------------------

The binary ONNX model is parsed without reading large ``raw_data`` payloads
or large external tensor payloads. Raw tensors strictly smaller than 128 bytes
are retained; tiny external tensors are loaded and inlined so shape inference
can use values such as a ``Reshape`` shape. Change this positive byte threshold
with ``--tiny-tensor-threshold BYTES``. Typed protobuf payload fields remain
part of the model metadata and are parsed normally.

Large external weights retain their original location, offset and length;
the command never rewrites their files. Large inline raw tensors cannot be
preserved after skipping their bytes: the command rejects such models before
saving instead of producing an incomplete model. Externalize those weights
first. This restriction also applies to tensor attributes and nested scopes.
Constant folding is disabled; only ``RepeatedLayer`` and the optimizer's normal
structural cleanup passes run.

``--output`` is required and must name a **new file**. Existing files, including
the input model and weight files, are never overwritten. When external references
remain, place the output in the input model's directory so relative paths remain
valid. Small, self-contained models can be written to another directory.
The original model and its external files must remain available when loading
the rewritten model's weights. Missing or invalid tiny shape tensors raise an
error rather than silently proceeding.

.. code-block:: bash

    python -m onnx_light layers model.onnx \
        --output model.layers.onnx \
        --tiny-tensor-threshold 256

For the complete argument reference, see
:doc:`../../api/command_lines/layers`.
