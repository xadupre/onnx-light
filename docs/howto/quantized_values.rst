.. _l-quantized-values:

Quantizes tensors into encoded values
====================================

The converters implement **portable onnx-light representations** of the
:ref:`quantization catalogue <l-next-steps-quantization>`, plus three explicit
:ref:`ORT MatMulNBits input layouts <l-ort-matmulnbits-inputs>`.
The portable profiles do not emit GGUF, Marlin or bitsandbytes buffers. Their profile
names identify numerical families, not those libraries' packing ABIs, published
bits-per-weight figures, training algorithms or accuracy guarantees.

``Tensor`` converts to an owned ``RuntimeValue`` of kind ``kEncoded``.
``TensorProto`` converts to ``EncodedValueProto``. Both use the same native
implementation and can be dequantized without the original plan. The returned
message has an inline ``StructTypeProto`` describing all fields and a versioned
native consumer identity. It can subsequently be put in a model's catalogue
and referenced by ``type_ref``.

The implementation lives in ``onnx_core``, not ``lib_onnx_proto``, and does not
require registered operator kernels. The Python module requires the runtime
bindings, as do other Python reference-runtime utilities.

Graph operators
---------------

``ai.rt::Quantize`` and ``ai.rt::Dequantize`` (opset 1) expose the native
codecs as CPU kernels, with ``LightOpSchema`` declarations and shape inference.
They are onnx-light extensions, not ONNX ``QuantizeLinear``/``DequantizeLinear``.

The :ref:`graph-kernel gallery example <l-example-quantization-kernels>`
demonstrates linear INT8 quantization with a scale and zero point, then nonlinear
NF4 codebook quantization with automatic or explicit scales. It also serializes
an encoded output for reuse as an initializer.

* ``Quantize(X, scales?, zero_points?, offsets?, codebooks?, permutation?,
  forward?, inverse?, outliers?) -> Y`` takes a floating tensor and returns an
  ``EncodedValueProto``. Its required ``type`` attribute is a ``TypeProto``
  containing the destination ``StructTypeProto``, inline or a model-catalogue
  reference. The encoded value retains the input's logical shape and dtype.
* ``Dequantize(X) -> Y`` takes an encoded value and returns a tensor.
  Its required integer ``dtype`` attribute selects ``FLOAT``, ``DOUBLE``,
  ``FLOAT16`` or ``BFLOAT16``. Conversion writes directly to that dtype and
  rejects nonfinite results and overflow.

``make_quantization_type(plan)`` returns a portable plan's storage descriptor
without requiring learned codebook values or populated transform matrices.
The type fixes block coverage and the sizes of tables, permutations and transforms;
it does not prescribe their numerical contents. For ORT layouts, use the
``struct_type`` of an existing ORT encoded value as the destination descriptor.
Its matrix dimensions, scale dtype and zero-point storage must match the new result.

When ``scales`` is omitted, ``Quantize`` calibrates **each block** from the
actual input, after outlier removal, permutation and the forward transform:
affine scales use the largest positive/negative ratio to the available code range,
scalar codebooks use their minimum/maximum entries, and cast blocks use scale one.
All-zero blocks use scale one. Default affine zero points are zero for signed
codes and the midpoint for unsigned codes; offsets default to zero. Values
outside a one-sided codebook/range still saturate or select the nearest entry.
ORT calibration follows column/K-block order and the existing source-dtype
rounding rules.

Explicit scales, zero points and offsets accept floating scalar tensors or
one-dimensional tensors with one value per block. They override calibration.
Codebooks concatenate all blocks' tables in run order. FLOAT, DOUBLE, FLOAT16
and BFLOAT16 parameter tensors are supported independently. Permutation and
outlier indices are one-dimensional INT64 tensors. Transform inputs contain
the declared square matrices in row-major order.

The kernel supplies the existing fixed scalar tables, but **does not train
learned codebooks or run GPTQ/AWQ optimization**. Learned codebooks must be
provided; vector/additive codebooks also require explicit scales. Nonempty
permutations, transforms and outlier indices must be supplied. Missing or
incompatible parameters fail explicitly. ORT layouts reject transforms,
outliers, codebooks and offsets.

For example, this creates a graph that calibrates INT4 blocks automatically:

.. code-block:: python

    from onnx_light import onnx
    from onnx_light.onnx import helper
    from onnx_light.onnx_core.quantization import (
        QuantizationFormat, make_quantization_plan, make_quantization_type,
    )

    plan = make_quantization_plan(QuantizationFormat.INT4, 8, block_size=4)
    destination = onnx.TypeProto()
    destination.struct_type.CopyFrom(make_quantization_type(plan))
    encode = helper.make_node("Quantize", ["X"], ["Q"], domain="ai.rt", type=destination)
    decode = helper.make_node(
        "Dequantize", ["Q"], ["Y"], domain="ai.rt", dtype=onnx.TensorProto.FLOAT,
    )
    graph = helper.make_graph(
        [encode, decode], "quantization",
        [helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [8])],
        [helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [8])],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("ai.rt", 1)],
    )

The runtime stores encoded edges in ``RuntimeContext.values()`` (Python
``get_value``/``put_value``), not in its ordinary tensor map. Sessions load
encoded initializers and resolve model-scoped types; child contexts inherit the
catalogue. Shape inference records Quantize's physical structured type.
Dequantize propagates a concrete encoded initializer's logical shape; otherwise
only its requested dtype is known.
These inference rules belong to ``onnx_core.shape_inference.infer_shapes_model``;
the ONNX-compatible ``onnx.shape_inference.infer_shapes`` does not register these
``ai.rt`` extension operators.
The model checker validates encoded initializer names, layouts and model-scoped
type references, including inside nested graphs.

Python
------

The runnable :ref:`Python profile tutorial <l-example-quantization-profiles>`
demonstrates **all 43 profiles**, including per-channel grouping, mixed
precision, supplied vector/additive codebooks, sparse outliers, rotations,
tiling and serialization. The examples use small explicit parameters; they
do not train or calibrate a model.

.. code-block:: python

    import numpy
    from onnx_light.onnx import numpy_helper
    from onnx_light.onnx_core.quantization import (
        QuantizationFormat,
        make_quantization_plan,
        quantize_tensor_proto,
        dequantize_tensor_proto,
    )

    weights = numpy.array([[-4, 0, 3.5], [-32, 0, 30]], dtype=numpy.float32)
    plan = make_quantization_plan(QuantizationFormat.EXL2, weights.size, block_size=3)
    runs = []
    for bits, scale in ((4, 0.5), (5, 2)):
        run = plan.run(0)
        run.layout.bits = bits
        block = run.block(0)
        block.scale = scale
        run.blocks = [block]
        runs.append(run)
    plan.runs = runs
    encoded = quantize_tensor_proto(numpy_helper.from_array(weights), plan)
    restored = numpy_helper.to_array(dequantize_tensor_proto(encoded))
    numpy.testing.assert_array_equal(restored, weights)

``plan.runs`` and ``run.blocks`` are converted to/from Python lists of copies.
``plan.run(i)`` returns one run copy; ``plan.set_run(i, run)`` replaces it.
Similarly, ``run.block(j)`` and ``run.set_block(j, block)`` access per-block
parameter copies. No Python object borrows a potentially invalidated vector element.
``quantize_tensor`` and ``dequantize_tensor`` accept/return the native runtime
``Tensor``. As with runtime structured feeds, Python represents an encoded
``RuntimeValue`` as ``EncodedValueProto`` rather than introducing another wrapper.

Choosing a profile
~~~~~~~~~~~~~~~~~~

Start with ``int8`` or ``int4`` for ordinary affine quantization, ``nf4`` for
a fixed nonlinear scalar table, or ``tiled_float`` for a floating-point cast.
Use ``int8_per_channel`` with explicit channel grouping when each channel needs
a different scale. Use ``exl2`` or ``exl3`` with explicit block overrides for
mixed bit widths. A named algorithm such as ``gptq`` or ``aqlm`` only selects
the representation defaults described in the catalogue below; it does not
apply that algorithm's optimization procedure.

``make_quantization_plan(format, count, block_size=128)`` divides ``count``
scalar elements into contiguous blocks of at most ``block_size`` elements.
It does not infer grouping from the source tensor shape. Both sizes are
element counts, not bytes or codebook-vector counts. For a NumPy array, pass
``array.size`` as ``count``. The last block may be shorter. Inputs are flattened
in logical row-major order, then reordered by any supplied permutation.
The factory creates one run for all full blocks and, if needed, a second run
for the shorter tail. An empty tensor has no runs.
The three ORT profiles instead use ``make_matmul_nbits_plan`` with explicit
matrix dimensions, as described below.

Pass a ``QuantizationFormat`` enum, for example ``QuantizationFormat.INT4``.
``quantization_format_name(format)`` returns its stable wire name;
``parse_quantization_format(name)`` converts an external string explicitly.
The factory and ``plan.format`` do not accept strings or integers.

Python parameter reference
~~~~~~~~~~~~~~~~~~~~~~~~~~

The returned ``QuantizationPlan`` is mutable. Its fields are:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Meaning
   * - ``format``
     - ``QuantizationFormat`` enum saved as a stable name in the encoded layout.
       Changing this enum alone
       does not reconfigure existing blocks; create a new plan to change defaults.
   * - ``runs``
     - List of ``QuantizationRun`` copies. Each run has one shared ``layout``
       and a nonempty list of per-block numerical parameters in ``blocks``.
       ``sum(run.layout.count * len(run.blocks) for run in plan.runs)`` must
       equal the number of source elements. Assign modified runs back to the plan.
       ORT profiles instead include padded elements at each column's tail.
   * - ``matrix_shape``
     - Required ``[K,N]`` source shape for ORT profiles, filled by their factory.
       Empty for portable profiles.
   * - ``permutation``
     - List of all flattened source indices, each exactly once, or an empty
       list for identity. Encoding gathers ``source[permutation[i]]``.
   * - ``transform_size``
     - Width of each row vector transformed before encoding; zero disables
       the transform. A nonzero width must divide the total element count.
   * - ``forward``, ``inverse``
     - Flat row-major lists, each containing ``transform_size ** 2`` numbers.
       Supply mutually inverse matrices, not only the forward rotation.
   * - ``outliers``
     - Unique flattened indices in the original source, before permutation.
       Values at these indices bypass quantization and are restored exactly.

Each run mirrors an array in ``StructTypeProto``. ``run.layout`` is a
``QuantizationBlockLayout`` holding the shared type parameters; ``run.blocks``
contains ``QuantizationBlockParameters`` with independent numerical values.
To change the layout of only some blocks, split them into separate runs.
Changing ``run.layout.bits`` changes the width for every block in that run.
All profiles initially use
``scale=1``, ``offset=0`` and ``zero_point=0``, except ``gptq``, ``awq`` and
``matmulnbits``, whose zero point defaults to 8.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Meaning and constraints
   * - ``run.layout.count``
     - Number of consecutive scalar elements covered by this block,
       at most ``UINT32_MAX``.
   * - ``run.layout.method``
     - ``QuantizationMethod.AFFINE``, ``CODEBOOK`` or ``CAST``.
       Changing it requires consistent parameters for the new method.
   * - ``run.layout.bits``
     - Code/index width, from 1 to 16. For codebooks, at least
       ``ceil(log2(entries))``. It is not necessarily the bits per source
       value for vector/additive books, and excludes metadata.
   * - ``run.layout.signed_codes``
     - Selects signed versus unsigned affine codes. A signed b-bit code
       ranges from ``-2**(b-1)`` to ``2**(b-1)-1``.
   * - ``block.scale``
     - Finite, strictly positive reconstruction multiplier, including for
       codebooks and casts. The explicit-plan APIs use it as supplied.
   * - ``block.zero_point``
     - Integer in the affine code range. Must be zero for codebooks and casts.
   * - ``block.offset``
     - Finite real additive offset for affine reconstruction; zero otherwise.
   * - ``run.layout.books``, ``entries``, ``vector_size``
     - Positive codebook dimensions. Defaults for each vector family appear
       in the catalogue below; scalar books have ``vector_size=1``.
   * - ``block.codebook``
     - Flat list of ``books * entries * vector_size`` finite numbers ordered
       by book, entry, then component. Must be empty for affine/cast blocks.
   * - ``run.layout.base3``
     - Packs five ternary indices per byte. Requires exactly one scalar
       three-entry codebook; set it to false for ordinary binary packing.
   * - ``run.layout.cast_type``
     - Physical dtype for ``CAST``: ``onnx.TensorProto.FLOAT``, ``DOUBLE``,
       ``FLOAT16`` or ``BFLOAT16``. Does not change the logical output dtype.
       Must be one of these four types for every method, since it is serialized
       in each portable block header even when unused.

For example, updating a single block without losing the change:

.. code-block:: python

    run = plan.run(0)
    block = run.block(0)
    block.scale = 0.25
    run.set_block(0, block)
    plan.set_run(0, run)

Do not write ``plan.runs[0].blocks[0].scale = 0.25`` and expect the plan to change:
the assignment only modifies a temporary copy.

Output, serialization and errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``quantize_tensor_proto(source, plan)`` returns an owned ``EncodedValueProto``.
``dequantize_tensor_proto(encoded, model=None)`` returns a ``TensorProto``
with the original logical shape and dtype, not the physical code dtype.
Convert it with ``numpy_helper.to_array``. The decoder does not need the
original plan; scales, tables, transforms and outliers are encoded with the data.
Proto conversions preserve ``name`` and ``doc_string`` presence independently:
absent fields remain absent, and explicitly empty fields remain present-empty.
The runtime ``Tensor`` API has only a plain string name and no documentation field.

Use ``encoded.SerializeToString()`` and
``onnx.EncodedValueProto().ParseFromString(...)`` to round-trip the message
through bytes. Keep its inline ``struct_type``, or save the model catalogue
and pass ``model=model`` when decoding a ``type_ref``. The tutorial includes
both forms. External tensor payloads must be loaded first.
Passing ``model=None`` explicitly is equivalent to omitting it, including for
``dequantize_tensor`` and ``export_matmul_nbits_inputs``.

Invalid parameters, missing codebooks/transforms, nonfinite values and
malformed payloads raise ``ValueError``. Affine values outside the code
range are clipped, not rejected: choose scales deliberately and measure
reconstruction error on representative data.
Encoding rejects codes whose reconstruction would be nonfinite or overflow
the logical output dtype. For portable profiles this check follows the inverse
transform, permutation and outlier restoration; wider intermediate values are
allowed when the final reconstruction fits. ORT profiles use the scales and
zero points rounded to the source dtype for this check.

An encoded message is not a drop-in tensor initializer for an ordinary
``MatMul`` or ``Attention``. Dequantize it first or supply an operator whose
schema and kernel explicitly support that representation.

C++
---

.. code-block:: cpp

    #include "onnx_core/runtime/quantization.h"

    using namespace onnx_light::core::runtime;
    Tensor input = Tensor::FromFloat("weight", {3}, {-4, 0, 3.5f});
    auto plan = MakeQuantizationPlan(QuantizationFormat::kInt4, 3);
    plan.runs[0].blocks[0].scale = 0.5;
    RuntimeValue encoded = QuantizeTensor(input, plan);
    Tensor restored = DequantizeTensor(encoded);

``MakeQuantizationType(plan)`` extracts the storage descriptor.
``QuantizeTensor(input, type, parameters, catalogue)`` applies the same
calibration as the graph operator; the overload taking a plan uses its
parameters unchanged.

``QuantizeTensorProto`` and ``DequantizeTensorProto`` provide the corresponding
message conversions. C++ dequantizers accept a ``StructTypeCatalogue`` for
model-scoped references; Python dequantizers accept an optional ``model``.
The C++ tensor dequantizer also accepts an allocator.
Its ``EncodedValueProto`` overload reads the message by const reference, without
first copying it into a ``RuntimeValue``. The Python tensor dequantizer uses
this overload too.

The converters allocate the final ``raw_data`` buffer at its exact size and
write into it directly. Proto encoding does not copy a completed encoded
message out of a runtime wrapper; proto decoding does not materialize an
intermediate output ``Tensor``. Numerical workspace (decoded values, transforms
and codebook search) is still allocated: these are reference codecs, not
zero-allocation conversions. Outputs own their storage independently of inputs.

``QuantizationFormats()`` is defined in the header as a ``constexpr`` function
returning ``std::array<QuantizationFormat, 43>`` without dynamic allocation:

.. code-block:: cpp

    constexpr auto formats = QuantizationFormats();
    static_assert(formats.front() == QuantizationFormat::kInt8);

The Python ``quantization_formats()`` function returns a list of
``QuantizationFormat`` values. C++ provides ``QuantizationFormatName`` and
``ParseQuantizationFormat`` for explicit conversions to and from stable wire names.

Numerical contract
------------------

This section describes the portable profiles. The ORT profiles have the
:ref:`matrix-specific contract below <l-ort-matmulnbits-inputs>`.

Inputs and logical outputs support FLOAT, DOUBLE, FLOAT16 and BFLOAT16.
Input shapes are concrete, including scalars and empty tensors. Inputs must
be finite. Source storage is never modified and results own their storage.
Nonfinite reconstructions and floating-point cast overflows are rejected.
External messages must have their payload loaded before conversion.
For a Python ``TensorProto``, call ``tensor.load_external_data(base_dir)``.
The loader preserves ``data_location=EXTERNAL`` and its file metadata;
quantization accepts it once ``raw_data`` is loaded, without changing that metadata.

Blocks cover the flattened tensor exactly, in order. Each can select:

* **Affine:** ``scale * (code - zero_point) + offset``. Codes have 1--16 bits
  and may be signed. Quantization clips to their range and rounds halfway
  cases to even, independently of the process rounding mode.
* **Codebook:** ``scale * sum(codebook[book, index, :])``. Tables have
  ``books * entries * vector_size`` doubles in row-major order. The encoder
  chooses each book's closest vector to the remaining residual; equal
  distances select the first entry. This is a deterministic reference
  encoder, not AQLM training or a global optimal additive-codebook search.
  A final partial vector compares only its logical components.
* **Cast:** a FLOAT, DOUBLE, FLOAT16 or BFLOAT16 scalar representation,
  with an optional multiplicative scale.

With the explicit-plan APIs, scales default to **one**, not to an
automatically estimated calibration.
Callers supply their scales, integer zero points, real offsets, trained
codebooks, rotations and selected outlier indices. Missing learned tables
and required rotations raise an error. No GPTQ Hessian calculation, AWQ
calibration, QAT or codebook training is performed.
Encoding rejects invalid enum values, including after a caller edits
``plan.format`` in C++. Python rejects assigning strings or integers to that
field. Decoding rejects unknown profile names in the encoded layout.

The plan applies these steps:

1. Saves selected outliers and substitutes zero before quantization.
2. Gathers ``permutation[i]`` into quantization position ``i``.
3. Applies ``forward`` to consecutive row vectors of ``transform_size`` values.
4. Encodes consecutive blocks.

Decoding applies ``inverse``, scatters back through the permutation, then
restores the original outliers. The supplied matrices must be finite square
inverse pairs (product within absolute tolerance ``1e-6``). This covers dense
rotations and diagonal rescaling; the matrices are not assumed orthogonal.
An empty permutation or transform is the identity. Per-channel quantization
and tiling are explicit: group the intended channel/tile values with a
permutation, choose matching block counts, and supply their parameters.

Catalogue coverage
------------------

``quantization_formats()`` returns these profiles. Parameters and block sizes
can be overridden: a profile supplies starting values, not a vendor schema.
In particular, hierarchical scale products are supplied as effective per-block
scales/offsets; this representation does not reproduce compressed scale layouts.

The third column contains executable plan configurations using this common
setup. Each row is independent. Loops show how to configure every profile in
the row; quantize inside the loop to use each resulting plan.
The scales and synthetic codebooks are illustrative, not calibrated or trained.
Replace them with your own parameters for real weights.

.. code-block:: python

    import numpy
    from onnx_light import onnx
    from onnx_light.onnx import numpy_helper
    from onnx_light.onnx_core.quantization import (
        QuantizationFormat,
        make_quantization_plan,
        quantize_tensor_proto,
        dequantize_tensor_proto,
    )

    weights = numpy.linspace(-1, 1, 16, dtype=numpy.float32).reshape(4, 4)
    n = weights.size

.. list-table::
   :header-rows: 1
   :widths: 20 35 45

   * - Profiles
     - Portable representation and caller inputs
     - Python plan configuration
   * - ``int8``, ``eetq``, ``int4``
     - Signed affine codes: 8 bits for the first two, 4 for ``int4``.
       Scales supplied explicitly.
     - .. code-block:: python

           for profile in (
               QuantizationFormat.INT8,
               QuantizationFormat.EETQ,
               QuantizationFormat.INT4,
           ):
               plan = make_quantization_plan(profile, n)
               run = plan.run(0)
               block = run.block(0)
               block.scale = 1 / (2 ** (run.layout.bits - 1) - 1)
               run.set_block(0, block)
               plan.set_run(0, run)

   * - ``int8_per_channel``
     - Signed INT8 with explicit channel grouping. Here columns are channels,
       gathered into blocks before encoding; decoding restores row-major order.
     - .. code-block:: python

           plan = make_quantization_plan(
               QuantizationFormat.INT8_PER_CHANNEL, n, block_size=weights.shape[0]
           )
           plan.permutation = numpy.arange(n).reshape(weights.shape).T.ravel().tolist()
           for i in range(weights.shape[1]):
               run = plan.run(0)
               block = run.block(i)
               maximum = float(numpy.abs(weights[:, i]).max())
               block.scale = maximum / 127 if maximum > 0 else 1
               run.set_block(i, block)
               plan.set_run(0, run)

   * - ``gptq``, ``awq``, ``matmulnbits``
     - Unsigned INT4 affine codes, default zero point 8; supplied group parameters.
     - .. code-block:: python

           for profile in (
               QuantizationFormat.GPTQ,
               QuantizationFormat.AWQ,
               QuantizationFormat.MATMULNBITS,
           ):
               plan = make_quantization_plan(profile, n, block_size=4)
               run = plan.run(0)
               blocks = run.blocks
               for block, scale in zip(blocks, [0.15, 0.05, 0.05, 0.15]):
                   block.scale = scale
                   block.zero_point = 8
               run.blocks = blocks
               plan.set_run(0, run)

   * - ``q2_k``, ``q3_k``, ``q4_k``, ``q5_k``, ``q6_k``
     - 2--6-bit affine blocks; supplied effective sub-block scales and offsets.
     - .. code-block:: python

           for profile in (
               QuantizationFormat.Q2_K,
               QuantizationFormat.Q3_K,
               QuantizationFormat.Q4_K,
               QuantizationFormat.Q5_K,
               QuantizationFormat.Q6_K,
           ):
               plan = make_quantization_plan(profile, n, block_size=4)
               run = plan.run(0)
               blocks = run.blocks
               for block in blocks:
                   block.scale = 1 / (2 ** (run.layout.bits - 1) - 1)
                   block.offset = 0.125
               run.blocks = blocks
               plan.set_run(0, run)

   * - ``hqq``, ``exl2``, ``exl3``
     - Signed 4-bit affine defaults; override bits, counts and parameters
       for mixed precision, or select a codebook block explicitly.
     - .. code-block:: python

           for profile in (
               QuantizationFormat.HQQ,
               QuantizationFormat.EXL2,
               QuantizationFormat.EXL3,
           ):
               plan = make_quantization_plan(profile, n, block_size=8)
               runs = []
               for bits, scale in ((3, 0.25), (5, 0.125)):
                   run = plan.run(0)
                   run.layout.bits = bits
                   block = run.block(0)
                   block.scale = scale
                   run.blocks = [block]
                   runs.append(run)
               plan.runs = runs

   * - ``nf4``, ``iq4_nl``
     - Fixed scalar codebooks and supplied scales. NF4 uses the full-precision
       normal-float table; IQ4_NL uses its 16 signed integer levels.
     - .. code-block:: python

           for profile in (QuantizationFormat.NF4, QuantizationFormat.IQ4_NL):
               plan = make_quantization_plan(profile, n)
               run = plan.run(0)
               block = run.block(0)
               block.scale = 1 if profile == QuantizationFormat.NF4 else 1 / 127
               run.set_block(0, block)
               plan.set_run(0, run)

   * - ``binary``
     - One-bit indices into ``[-1, 1]``.
     - .. code-block:: python

           plan = make_quantization_plan(QuantizationFormat.BINARY, n)
           run = plan.run(0)
           block = run.block(0)
           block.scale = 0.5
           run.set_block(0, block)
           plan.set_run(0, run)

   * - ``ternary``, ``tq1_0``, ``bitnet``, ``paretoq``, ``tequila``
     - Indices into ``[-1, 0, 1]``; five base-3 digits per byte.
       This represents ternary weights, not the associated training procedures.
     - .. code-block:: python

           for profile in (
               QuantizationFormat.TERNARY,
               QuantizationFormat.TQ1_0,
               QuantizationFormat.BITNET,
               QuantizationFormat.PARETOQ,
               QuantizationFormat.TEQUILA,
           ):
               plan = make_quantization_plan(profile, n)
               run = plan.run(0)
               block = run.block(0)
               block.scale = 0.75
               run.set_block(0, block)
               plan.set_run(0, run)

   * - ``tq2_0``
     - The same ternary table with two bits per index.
     - .. code-block:: python

           plan = make_quantization_plan(QuantizationFormat.TQ2_0, n)
           run = plan.run(0)
           block = run.block(0)
           block.scale = 0.75
           run.set_block(0, block)
           plan.set_run(0, run)

   * - ``stq1_0``
     - Supplied vector codebook; defaults to 32 four-component entries and 5-bit
       indices. Code/sign splitting and vendor scatter layouts are not emitted.
     - .. code-block:: python

           plan = make_quantization_plan(QuantizationFormat.STQ1_0, n)
           run = plan.run(0)
           block = run.block(0)
           table = numpy.linspace(-1, 1, 128).reshape(1, 32, 4)
           block.codebook = table.ravel().tolist()
           run.set_block(0, block)
           plan.set_run(0, run)

   * - ``iq1_s``
     - Supplied vector codebooks; defaults to 256 eight-component entries.
     - .. code-block:: python

           plan = make_quantization_plan(QuantizationFormat.IQ1_S, n)
           run = plan.run(0)
           block = run.block(0)
           table = numpy.linspace(-1, 1, 2048).reshape(1, 256, 8)
           block.codebook = table.ravel().tolist()
           run.set_block(0, block)
           plan.set_run(0, run)

   * - ``quip_sharp``
     - The same default table dimensions as ``iq1_s``, plus an explicit
       transform pair. This example uses a two-component orthogonal rotation.
     - .. code-block:: python

           plan = make_quantization_plan(QuantizationFormat.QUIP_SHARP, n)
           run = plan.run(0)
           block = run.block(0)
           table = numpy.linspace(-1, 1, 2048).reshape(1, 256, 8)
           block.codebook = table.ravel().tolist()
           run.set_block(0, block)
           plan.set_run(0, run)
           rotation = numpy.array([[1, 1], [1, -1]]) / numpy.sqrt(2)
           plan.transform_size = 2
           plan.forward = rotation.ravel().tolist()
           plan.inverse = rotation.T.ravel().tolist()

   * - ``aqlm``
     - Supplied additive vector codebooks; defaults to two 256-entry,
       eight-component books with 8-bit indices.
     - .. code-block:: python

           plan = make_quantization_plan(QuantizationFormat.AQLM, n)
           run = plan.run(0)
           block = run.block(0)
           table = numpy.empty((2, 256, 8))
           table[0] = numpy.linspace(-1, 1, 256)[:, None]
           table[1] = numpy.linspace(-0.125, 0.125, 256)[:, None]
           block.codebook = table.ravel().tolist()
           run.set_block(0, block)
           plan.set_run(0, run)

   * - ``spqr``
     - Signed 4-bit affine base plus exact sparse outliers. Indices refer to
       the original flattened tensor; selection is the caller's responsibility.
     - .. code-block:: python

           plan = make_quantization_plan(QuantizationFormat.SPQR, n)
           plan.outliers = [0, 15]
           run = plan.run(0)
           block = run.block(0)
           block.scale = 0.125
           run.set_block(0, block)
           plan.set_run(0, run)

   * - ``squeezellm``
     - Supplied scalar-codebook base plus exact sparse outliers.
       Defaults to 16 supplied levels.
     - .. code-block:: python

           plan = make_quantization_plan(QuantizationFormat.SQUEEZELLM, n)
           plan.outliers = [0, 15]
           run = plan.run(0)
           block = run.block(0)
           block.codebook = numpy.linspace(-1, 1, 16).tolist()
           run.set_block(0, block)
           plan.set_run(0, run)

   * - ``log``
     - Scalar codebook with zero and signed powers of two from ``2**-3`` through
       ``2**3``. Supply another table for a different base, range or logarithmic rule.
     - .. code-block:: python

           plan = make_quantization_plan(QuantizationFormat.LOG, n)
           run = plan.run(0)
           block = run.block(0)
           block.scale = 0.125
           run.set_block(0, block)
           plan.set_run(0, run)

   * - ``fp6_llm``, ``mxfp6``
     - E3M2 finite levels and supplied scales, using 6-bit codebook indices.
     - .. code-block:: python

           for profile in (QuantizationFormat.FP6_LLM, QuantizationFormat.MXFP6):
               plan = make_quantization_plan(profile, n)
               run = plan.run(0)
               block = run.block(0)
               block.scale = 0.25
               run.set_block(0, block)
               plan.set_run(0, run)

   * - ``mxfp4``, ``nvfp4``
     - E2M1 finite levels and supplied effective scales. The caller rounds scales
       to E8M0/FP8 and combines scale levels if required by their numerical profile.
     - .. code-block:: python

           for profile in (QuantizationFormat.MXFP4, QuantizationFormat.NVFP4):
               plan = make_quantization_plan(profile, n, block_size=8)
               run = plan.run(0)
               blocks = run.blocks
               for block, scale in zip(blocks, [0.25, 0.5]):
                   block.scale = scale
               run.blocks = blocks
               plan.set_run(0, run)

   * - ``fp8_e4m3``
     - Finite E4M3FN levels with 8-bit codebook indices; nonfinite levels excluded.
     - .. code-block:: python

           plan = make_quantization_plan(QuantizationFormat.FP8_E4M3, n)
           run = plan.run(0)
           block = run.block(0)
           block.scale = 0.5
           run.set_block(0, block)
           plan.set_run(0, run)

   * - ``quarot``, ``smoothquant``
     - Signed affine blocks (4 and 8 bits respectively), with explicit
       forward/inverse rotations or rescaling.
     - .. code-block:: python

           for profile in (QuantizationFormat.QUAROT, QuantizationFormat.SMOOTHQUANT):
               plan = make_quantization_plan(profile, n)
               matrix = (
                   numpy.array([[1, 1], [1, -1]]) / numpy.sqrt(2)
                   if profile == QuantizationFormat.QUAROT
                   else numpy.diag([2.0, 0.5])
               )
               plan.transform_size = 2
               plan.forward = matrix.ravel().tolist()
               plan.inverse = numpy.linalg.inv(matrix).ravel().tolist()
               run = plan.run(0)
               block = run.block(0)
               block.scale = 0.25
               run.set_block(0, block)
               plan.set_run(0, run)

   * - ``tiled_float``
     - FLOAT casts by default. Here an explicit permutation groups 2-by-2
       tiles and the physical storage is changed to FLOAT16.
     - .. code-block:: python

           plan = make_quantization_plan(
               QuantizationFormat.TILED_FLOAT, n, block_size=4
           )
           indices = numpy.arange(n).reshape(4, 4)
           plan.permutation = (
               indices.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).ravel().tolist()
           )
           run = plan.run(0)
           run.layout.cast_type = onnx.TensorProto.FLOAT16
           plan.set_run(0, run)

   * - ``column_major``
     - FLOAT casts with a supplied column-major ordering permutation.
       Decoding restores the original logical shape and order.
     - .. code-block:: python

           plan = make_quantization_plan(QuantizationFormat.COLUMN_MAJOR, n)
           plan.permutation = numpy.arange(n).reshape(weights.shape).T.ravel().tolist()

After any row, use its configured plan as follows (or put these lines inside
the profile loop to encode each profile):

.. code-block:: python

    encoded = quantize_tensor_proto(numpy_helper.from_array(weights), plan)
    restored = numpy_helper.to_array(dequantize_tensor_proto(encoded))
    print(plan.format, numpy.max(numpy.abs(restored - weights)))

All floating-point/codebook profiles use closest-level encoding, with first-entry
ties. They do not promise the external format's float bit patterns or tie-breaking.
Scale tensors and learned codebooks belong to payload storage, not shared type
constants. Consecutive blocks with the same physical layout share one array
element declaration; changing scales or table contents does not duplicate the
descriptor. This reference representation prioritizes correctness and explicit
semantics over minimal payload size or fast quantization.

Wire layout and validation
--------------------------

This section describes the portable wire layout. The ORT profiles use the
B/scales/zero_points layout described below instead.

The structured root name is ``onnx_light.quantization.v1/<profile>``. Its fields
are, in order: one zero reserved byte, an INT64 permutation, DOUBLE forward and
inverse matrices, INT64 outlier indices, DOUBLE outlier values, and a structure
of block runs with a constant total block count. Each run is an array of blocks
with identical layout parameters, named ``run_<first-block-index>``. Adjacent
compatible runs are merged. Integers/floats in the payload are little-endian.
``QuantizationPlan.runs`` follows this same organization in memory, with one
layout per run instead of duplicating it in every block. The enum is converted
to the existing profile name; neither the versioned wire schema nor the
per-block payload order changes.

Each block has a nine-element INT64 type constant containing count, method,
index width, signedness, number of books, entries, vector width, base-3 flag and
cast dtype. Method values are affine=0, codebook=1 and cast=2.
Instance fields are DOUBLE scale, DOUBLE zero point, DOUBLE offset,
the DOUBLE codebook and UINT8 packed codes. Binary indices are LSB-first, with
zero high padding bits; base-3 packing stores the first index in the least
significant trit and zero unused high trits.
For additive codebooks, indices are ordered by logical vector, then book;
each table is ordered by book, entry, then vector component. A partial final
vector still stores one index per book. Cast codes are the requested floating
dtype's ordinary little-endian bytes.

The descriptor is checked against this exact versioned schema, including field
names/types/dimensions. The generic proto validator checks payload extent and
layout first; the converter additionally checks parameters, codebook indices,
padding, permutations, inverse matrices and exact logical coverage. Unknown
layouts are rejected, never interpreted by shape or profile name alone.
Only this native consumer is implemented: the descriptor is not an automatically
executable ONNX ``FunctionProto`` decoder, and tensor-only operators cannot consume
it without explicit dequantization or a matching custom kernel.

.. _l-ort-matmulnbits-inputs:

ONNX Runtime MatMulNBits inputs
-------------------------------

``QuantizationFormat.ORT_MATMULNBITS_INT2``, ``ORT_MATMULNBITS_INT4`` and
``ORT_MATMULNBITS_INT8`` implement the input packing of
``com.microsoft::MatMulNBits`` version 1. They are separate from the original
``MATMULNBITS`` profile, which remains a portable onnx-light affine codec.
These formats do **not** implement CPU microkernel or CUDA ``weight_prepacked``
layouts; the execution provider may still prepack the exported inputs internally.

``make_matmul_nbits_plan(format, k, n, block_size=128)`` accepts a positive
``[K,N]`` matrix shape and a power-of-two block size of at least 16 (at most
``UINT32_MAX``). Execution providers may restrict this further; the ORT CPU
implementation supports 16, 32, 64, 128 and 256. The plan records
``matrix_shape=[K,N]`` and rejects mismatched sources.
Sources are FLOAT, FLOAT16 or BFLOAT16 matrices, not DOUBLE or arbitrary-rank tensors.
BFLOAT16 kernel availability depends on the ORT execution provider and version.
ORT CPU currently also lacks the 8-bit unpacked-compute path selected by floating
zero points. For INT8 execution on that provider, use implicit or packed integer
zero points; floating zero points remain supported by this codec and the operator schema.

There is one shared run layout and ``N * ceil(K/block_size)`` parameter blocks,
ordered by column first, then by group along K. Each layout count is the full
block size, including padding. Scales default to one; zero points default to
``2**(bits-1)``. Supply parameters explicitly; this is packing and conversion,
not an implementation of ORT's scale-calibration algorithm.

Scales and floating zero points are rounded to the source dtype **before**
quantization so decoding and ORT use the same values. Codes are unsigned and
use nearest-even rounding and clipping to ``[0, 2**bits-1]``. The reconstruction
is ``(code-zero_point)*scale``. Negative finite scales are accepted. A zero
scale is allowed only for an all-zero source group, and nonzero scales that
round to zero are rejected. Offsets, codebooks, permutations, transforms,
outliers, ``g_idx`` and fused bias are not part of these representations.

The encoded root name is
``onnx_light.quantization.v1/ort_matmulnbits_int{2,4,8}``. Its logical type is
the original ``[K,N]`` matrix. Its fields are:

.. list-table::
   :header-rows: 1

   * - Field
     - Type and shape
   * - ``parameters``
     - INT64 type constant ``[bits, block_size]``; consumes no payload bytes.
   * - ``B``
     - UINT8 ``[N, ceil(K/block_size), block_size*bits/8]``.
   * - ``scales``
     - Source dtype ``[N, ceil(K/block_size)]``.
   * - ``zero_points`` (optional)
     - UINT8 ``[N, ceil(ceil(K/block_size)*bits/8)]`` for packed integer zero
       points, or source dtype ``[N, ceil(K/block_size)]`` for floating zero points.

``raw_data`` concatenates B, scales and optional zero points, with no portable
codec header or per-block DOUBLE metadata between them. B codes are packed
least-significant bits first within each K block. The last block of each
column is padded with zero codes. Packed zero points restart at a byte
boundary for every column; unused high bits are zero.
Both decoding and export reject nonzero weight-tail codes or unused zero-point bits.
If all effective zero points equal the midpoint, the zero-point tensor is
omitted. Otherwise all in-range integer zero points use packed UINT8 storage;
any fractional or out-of-range value selects floating storage for the whole tensor.

``export_matmul_nbits_inputs(encoded, model=None)`` validates the descriptor
and extracts owned TensorProto inputs without dequantizing. Its result exposes
``weights``, ``scales``, optional ``zero_points`` (``None`` when implicit),
and the attributes ``k``, ``n``, ``bits`` and ``block_size``.
Use the tensors as initializers for a normal ``MatMulNBits`` node, leaving
``weight_prepacked`` unset. Inline layouts and model-catalogue references both work.

``plan.matrix_shape`` is the native Python-visible ``Shape`` holding ``[K, N]``,
also exported by ``onnx_light.onnx_core.shape_inference`` and
``onnx_light.onnx_core.quantization``. Its getter returns a mutable view that
keeps the plan alive; assignment accepts a ``Shape`` or an integer list/tuple
and copies the dimensions. Use ``list(plan.matrix_shape)`` for a plain list.
Changing this shape does not rebuild the plan's blocks; incompatible geometry
is rejected when encoding.

.. code-block:: python

    import numpy
    from onnx_light.onnx import helper, numpy_helper
    from onnx_light.onnx_core.quantization import (
        QuantizationFormat,
        make_matmul_nbits_plan,
        quantize_tensor_proto,
        export_matmul_nbits_inputs,
    )

    weights = (numpy.arange(35 * 3).reshape(35, 3) % 3 - 1).astype(numpy.float32)
    plan = make_matmul_nbits_plan(
        QuantizationFormat.ORT_MATMULNBITS_INT4, 35, 3, block_size=16
    )
    encoded = quantize_tensor_proto(numpy_helper.from_array(weights), plan)
    inputs = export_matmul_nbits_inputs(encoded)
    initializers = [inputs.weights, inputs.scales]
    names = ["A", inputs.weights.name, inputs.scales.name]
    if inputs.zero_points is not None:
        initializers.append(inputs.zero_points)
        names.append(inputs.zero_points.name)
    node = helper.make_node(
        "MatMulNBits",
        names,
        ["Y"],
        domain="com.microsoft",
        K=inputs.k,
        N=inputs.n,
        bits=inputs.bits,
        block_size=inputs.block_size,
    )

The C++ equivalents are ``MakeMatMulNBitsPlan`` and ``ExportMatMulNBitsInputs``.
The contract follows the `ORT operator schema
<https://github.com/microsoft/onnxruntime/blob/ee5f6e7cfa1f9e4e4253154e7c719a846280ca83/docs/ContribOperators.md#com.microsoft.MatMulNBits>`_
and `weight quantizer input shapes
<https://github.com/microsoft/onnxruntime/blob/ee5f6e7cfa1f9e4e4253154e7c719a846280ca83/onnxruntime/python/tools/quantization/matmul_nbits_quantizer.py>`_.
