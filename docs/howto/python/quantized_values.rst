.. _l-quantized-values:
.. _l-quantized-values-python:

Quantizes tensors into encoded values
=====================================

This guide covers the Python interface. For the other language, see
:doc:`../cpp/quantized_values`.

The converters implement **portable onnx-light representations** of the
:ref:`quantization catalogue <l-next-steps-quantization>`, plus three explicit
:ref:`ORT MatMulNBits input layouts <l-ort-matmulnbits-inputs-python>`.
The portable profiles do not emit GGUF, Marlin or bitsandbytes buffers. Their profile
names identify numerical families, not those libraries' packing ABIs, published
bits-per-weight figures, training algorithms or accuracy guarantees.

``Tensor`` converts to an owned ``RuntimeValue`` of kind ``kEncoded``.
``TensorProto`` converts to ``EncodedValueProto``. Both use the same native
implementation and can be dequantized without the original plan. A self-contained
message has an inline ``StructTypeProto`` describing all fields and a versioned
native consumer identity. It can subsequently be put in a model's catalogue
and referenced by ``type_ref``.

The implementation lives in ``onnx_core``, not ``lib_onnx_proto``, and does not
require registered operator kernels. The Python module requires the runtime
bindings, as do other Python reference-runtime utilities.

.. _l-mxfp6-e2m3-weights:

Complete MXFP6 E2M3 profile
-------------------------

``QuantizationFormat.MXFP6_E2M3`` (``mxfp6_e2m3``) implements the
compressed-tensors E2M3/group-32/E8M0 weight contract. The existing ``mxfp6``
and ``fp6_llm`` profiles remain **E3M2 codebooks with caller-supplied scales**:
their names, schemas and payload decoding are unchanged.

The reference is `compressed-tensors revision
eb058d630f7344099459b6795b1e3eeffdda4b40
<https://github.com/vllm-project/compressed-tensors/blob/eb058d630f7344099459b6795b1e3eeffdda4b40/src/compressed_tensors/compressors/mxfp6/base.py>`_,
including ``mxfp6/helpers.py``, ``quantization/utils/fp6_utils.py`` and
``quantization/utils/mxfp_utils.py``:

* Input is a finite FLOAT32 matrix ``[N,K]``, with positive dimensions and
  ``K`` divisible by 32. Groups are consecutive columns on axis 1; they never
  cross rows. Incomplete groups, empty matrices, other dtypes and nonfinite
  inputs are rejected, without padding or replacement.
* Each code uses sign bit 5, exponent bits 4:3 with bias 1 and mantissa bits
  2:0. Exponent zero has magnitude ``mantissa/8``; otherwise magnitude is
  ``(1+mantissa/8)*2**(exponent-1)``. All 64 codes are finite; both signed
  zeros are preserved, and maximum magnitude is 7.5.
* Four consecutive codes form
  ``c0 | (c1 << 6) | (c2 << 12) | (c3 << 18)``, stored as three little-endian
  bytes. For example, upstream's ``[1,-0.125,7.5,0]`` fixture has codes
  ``[8,33,31,0]`` and bytes ``48 f8 01``.
* Scale selection computes
  ``rounded_bits = (float32_bits(max_abs) + (1 << 19)) & 0x7f800000``.
  The scale exponent is the resulting exponent minus 127 minus 2, clamped
  to ``[-127,125]``. For normal values this increments ``floor(log2(max_abs))``
  at mantissa 1.9375, then subtracts 2; it is not generic nearest-power-of-two
  rounding. Zero blocks use exponent -127.
* E8M0 bytes encode ``scale = 2**(byte-127)``. Byte 0 is ``2**-127``, not
  zero; byte 255 is rejected. Scale one is byte 127. The upper exponent
  clamp is an explicit onnx-light extension ensuring finite reconstruction
  even for maximum finite FLOAT32 inputs, not an upstream parity guarantee
  at extreme ranges.
* Normalization divides in FLOAT32, saturates to ``[-7.5,7.5]`` and converts
  to E2M3 using nearest-even rounding, including subnormal and exponent-boundary
  ties. Reconstruction multiplies the decoded E2M3 value by its scale in
  FLOAT32. Upstream's decompressor returns BFLOAT16; the finite E2M3-times-E8M0
  values agree numerically, while this API returns FLOAT32.

``make_quantization_plan(QuantizationFormat.MXFP6_E2M3, count)`` selects
automatic scales, fixed groups of 32 and FLOAT32 reconstruction.
``count`` must be a positive multiple of 32; ``block_size`` accepts 32 or
the default 128, both selecting 32. Modified layouts, supplied scales,
transforms, permutations, outliers, shared parameters and
``make_quantization_type`` are rejected.

.. code-block:: python

    import numpy
    import onnx_light.onnx.numpy_helper as onh
    from onnx_light.onnx_core import quantization

    weights = numpy.tile([1, -0.125, 7.5, 0], 8).astype(numpy.float32).reshape(1, 32)
    plan = quantization.make_quantization_plan(
        quantization.QuantizationFormat.MXFP6_E2M3, weights.size
    )
    encoded = quantization.quantize_tensor_proto(onh.from_array(weights), plan)
    parameters = quantization.export_mxfp6_e2m3_weights(encoded)
    encoded_again = quantization.import_mxfp6_e2m3_weights(parameters)
    restored = onh.to_array(quantization.dequantize_tensor_proto(encoded_again))

``quantize_mxfp6_e2m3_weights`` also accepts a runtime ``Tensor`` directly.
``Mxfp6E2m3Weights`` owns UINT8 ``weight_packed`` shaped ``[N,3*K/4]`` and
UINT8 ``weight_scale`` shaped ``[N,K/32]``. Export/import preserves these bytes
without requantization. Imports accept scale bytes 0--254 only if every
reconstruction remains finite. Incorrect dtypes, shapes, payload sizes and
schema constants are rejected.

The self-describing ``onnx_light.mxfp6_e2m3_weights.v1`` schema records the
code layout, group size, axis, scale selection, rounding, saturation,
reconstruction dtype and rejection policies. Its payload contains row-major
packed weights followed by row-major scale bytes. It supports inline schemas
and model catalogue references, without retaining a quantization plan.
This is not a general checkpoint importer or accelerated matrix multiplication.

.. _l-mxfp8-weights:

Complete MXFP8 profile
---------------------

``QuantizationFormat.MXFP8`` (``mxfp8``) implements E4M3FN values with
automatically selected E8M0 scales for groups of 32. Unlike the existing
``fp8_e4m3`` portable codebook profile, it stores native FP8 codes, not
codebook indices, and does not require caller-prepared effective scales.
The existing ``fp8_e4m3`` contract is unchanged.

The reference is `compressed-tensors revision
eb058d630f7344099459b6795b1e3eeffdda4b40
<https://github.com/vllm-project/compressed-tensors/blob/eb058d630f7344099459b6795b1e3eeffdda4b40/src/compressed_tensors/compressors/mxfp8/base.py>`_,
together with its ``quantization/utils/mxfp_utils.py`` scale-selection and
``compressors/mx_utils.py`` scale-storage helpers:

* Input is a finite FLOAT32 matrix ``[N,K]`` with positive dimensions and
  ``K`` divisible by 32. Groups are consecutive columns on axis 1, never
  across rows. Incomplete groups, empty matrices, other dtypes and
  nonfinite inputs are rejected rather than padded or replaced.
* For a nonzero group maximum ``m``, the pinned FLOAT32 helper computes
  ``rounded_bits = (float32_bits(m) + (1 << 21)) & 0x7f800000``.
  The scale exponent is the resulting exponent minus 127 minus 8.
  For normal inputs this increments ``floor(log2(m))`` at a mantissa
  threshold of 1.75, then subtracts 8. This deliberately uses the pinned
  helper's E2M1 threshold, not a generic nearest-power-of-two or ceiling rule.
* This profile clamps scale exponents to ``[-127,119]`` and uses scale one
  for all-zero groups. These are explicit onnx-light policies for zero
  blocks and extreme ranges, not upstream parity guarantees there.
  Scales are positive powers of two, encoded as UINT8 exponents with bias
  127; byte zero denotes ``2**-127``, not zero. Byte ``0xff`` is invalid.
* Normalization divides in FLOAT32, saturates to ``[-448,448]`` and converts
  to native E4M3FN with round-to-nearest-even, including subnormal ties.
  Signed zero is preserved. Reconstruction multiplies the decoded E4M3FN
  value by the decoded E8M0 scale in FLOAT32.
* Imported scale bytes 0--254 are accepted only when every reconstructed
  value remains finite. FP8 NaN codes, E8M0 NaN, nonfinite reconstruction,
  incorrect dtypes, shapes, payload sizes and schema constants are rejected.

``make_quantization_plan(QuantizationFormat.MXFP8, count)`` selects this
automatic contract; grouping and reconstruction are fixed, not customizable
portable codebook parameters. ``count`` must be a positive multiple of 32;
``block_size`` accepts 32 or the factory default 128, both selecting groups
of 32. Reconstruction is always FLOAT32. Custom scales, transforms,
permutations, outliers, shared parameters and ``make_quantization_type``
are not supported for this profile.
The dedicated ``quantize_mxfp8_weights`` entry
point implements the same codec directly:

.. code-block:: python

    import numpy
    import onnx_light.onnx.numpy_helper as onh
    from onnx_light.onnx_core import quantization

    weights = numpy.ones((2, 64), dtype=numpy.float32)
    plan = quantization.make_quantization_plan(
        quantization.QuantizationFormat.MXFP8, weights.size
    )
    encoded = quantization.quantize_tensor_proto(onh.from_array(weights), plan)
    parameters = quantization.export_mxfp8_weights(encoded)
    encoded_again = quantization.import_mxfp8_weights(parameters)
    restored = onh.to_array(quantization.dequantize_tensor_proto(encoded_again))

``Mxfp8Weights`` owns ``weight`` (FLOAT8E4M3FN ``[N,K]``) and
``weight_scale`` (UINT8 ``[N,K/32]``), matching the upstream parameter roles.
Export/import preserves their bytes exactly without requantization.
The self-describing ``onnx_light.mxfp8_weights.v1`` descriptor records
the shape, axis, group size, dtypes, rounding, saturation and invalid-input
policies. Its raw payload contains row-major FP8 bytes followed by row-major
E8M0 bytes, with no padding or hardware swizzle. Serialize the complete
``EncodedValueProto``, not just ``raw_data``; both dequantizers work without
the original plan. Referenced types also require their model catalogue.

Compatibility covers scale selection, codes, scale bytes and FLOAT32
reconstruction for finite complete groups with a normal, nonzero maximum
within the unclamped scale range.
The upstream decompressor casts scales to BFLOAT16 and commonly reconstructs
BFLOAT16 weights; this profile instead promises FLOAT32 reconstruction,
not universal bitwise equality to upstream output dtypes. It is not a general
checkpoint importer, a claim of every vendor's binary layout, activation
calibration, or **accelerated native MXFP8 matrix multiplication**.
The arithmetic assumes IEEE nearest-even, gradual underflow and no fast-math.

.. _l-mxfp4-weights:

Complete MXFP4 weights
----------------------

``quantize_mxfp4_weights`` implements the complete fixed-weight codec
``onnx_light.mxfp4_weights.v1``. It is separate from the existing
``QuantizationFormat.MXFP4`` (``mxfp4``) portable numerical profile, whose
parameters and serialized decoder are unchanged.

The numerical contract follows the group-size-32 MXFP4 convention at
`compressed-tensors revision eb058d630f7344099459b6795b1e3eeffdda4b40
<https://github.com/vllm-project/compressed-tensors/tree/eb058d630f7344099459b6795b1e3eeffdda4b40/src/compressed_tensors/compressors/mxfp4>`_,
including its MX scale helpers and E2M1 rounding. The complete contract is:

* Input is a finite FLOAT matrix ``[N,K]``. Groups are consecutive columns on
  axis 1; ``K`` must be a positive multiple of 32. Incomplete groups, empty
  matrices and nonfinite inputs are rejected, not padded.
* For each nonzero group, ``round_to_power_2(max(abs(group)))`` is computed
  using the pinned helper's E2M1 mantissa threshold, then divided by four
  (the E2M1 exponent offset of 2) to obtain the scale. An all-zero group uses
  scale one. The scale exponent is clamped to ``[-127, 125]`` to keep FLOAT
  reconstruction finite. The stored UINT8 scale is the biased exponent (bias
  127); ``0xff`` is invalid.
* Values are divided by the selected scale in FLOAT32, saturated to ``[-6,6]``,
  then rounded to E2M1 with midpoint ties to even. Negative normalized values
  that round to zero retain the sign bit; input ``-0.0`` and normalized ``-0.0``
  encode as positive zero.
* E2M1 sign is bit 3 and magnitude is bits 0--2. The first column occupies the
  low nibble. Payload bytes contain row-major packed values followed by row-major
  scale bytes. This byte ordering is explicit and swizzle-free, but does not
  claim compatibility with every vendor checkpoint packing ABI.

The codec supplies numerical and portable storage compatibility for this
contract. It does not provide an accelerated MXFP4 compute kernel.

.. code-block:: python

    import numpy
    from onnx_light import onnx
    from onnx_light.onnx_core import quantization
    from onnx_light.onnx_py._onnxpykernels import runtime

    weights = runtime.tensor_from_numpy(
        "W", onnx.TensorProto.FLOAT, [2, 64],
        numpy.ones((2, 64), dtype=numpy.float32).view(numpy.uint8).reshape(-1),
    )
    encoded = quantization.quantize_mxfp4_weights(weights)
    parameters = quantization.export_mxfp4_weights(encoded)
    encoded_again = quantization.import_mxfp4_weights(parameters)

``Mxfp4Weights`` owns UINT8 ``weight_packed`` (``[N,K/2]``) and
``weight_scale`` (``[N,K/32]``). Export/import preserves the packed nibbles
and E8M0 bytes exactly. The self-describing encoded value survives ordinary
wire serialization and is decoded by both ``dequantize_tensor`` and
``dequantize_tensor_proto``.

.. _l-nvfp4-weights:

Complete NVFP4 weights
----------------------

``quantize_nvfp4_weights`` implements a **separate** complete weight contract,
``onnx_light.nvfp4_weights.v1``. The existing ``QuantizationFormat.NVFP4``
(``nvfp4``) remains the portable E2M1 numerical profile with caller-supplied
effective scales; its payload and decoding are unchanged.

The complete contract follows the symmetric, group-size-16 weight layout in
`compressed-tensors at eb058d630f7344099459b6795b1e3eeffdda4b40
<https://github.com/vllm-project/compressed-tensors/blob/eb058d630f7344099459b6795b1e3eeffdda4b40/src/compressed_tensors/compressors/nvfp4/base.py>`_,
including ``nvfp4/helpers.py``, ``quantization/lifecycle/forward_helpers.py``
and the NVFP4 weight presets at that revision. It uses the FLOAT32 arithmetic
path, not the compressor's default BFLOAT16 decompression output.
The presets round calibrated local scales to FP8 before compression. In contrast,
calling the upstream compressor directly with arbitrary unrounded FLOAT scales
quantizes before its storage cast; this codec intentionally does not reproduce
that inconsistent parameter combination.

* Weights are a nonempty, row-major FLOAT matrix ``[N, K]``. Groups run along
  axis 1, never across rows; ``K`` must be divisible by 16. Incomplete groups,
  empty matrices and nonfinite weights are rejected, not padded or replaced.
* The caller supplies local scales ``[N, K/16]`` as FLOAT or FLOAT8E4M3FN and
  one finite, strictly positive FLOAT32 global scale. This is a fixed-parameter
  codec; it does not perform calibration.
* FLOAT local scales are rounded **first** to E4M3FN, nearest-even, with finite
  saturation at 448. Local scales must be strictly positive before and after
  rounding. Thus an all-zero weight group still requires a positive local scale;
  zero scales, negative scales, NaNs, and values rounding to zero are rejected.
* For each group, the effective reconstruction scale is
  ``float32(float32(local_fp8) / global_scale)``. It must remain finite and
  strictly positive. Each weight is divided by that rounded scale in FLOAT32,
  saturated to ``[-6, 6]``, then rounded to E2M1, nearest-even.
  The magnitude levels are ``[0, .5, 1, 1.5, 2, 3, 4, 6]``.
  Negative normalized values rounding to E2M1 zero retain a negative-zero code.
  An input ``-0.0``, or normalization underflow to ``-0.0``, encodes as positive
  zero, following upstream's strict ``normalized < 0`` sign test.
* Codes use bit 3 for sign and bits 0--2 for magnitude. Consecutive columns
  occupy the low then high nibble of each byte. There is no hardware swizzle.
  Reconstruction multiplies the decoded E2M1 value by the rounded effective
  scale in FLOAT32. Nonfinite reconstructions are rejected.

These rules assume IEEE round-to-nearest-even, gradual underflow and no
fast-math. They do **not** substitute an arbitrary precombined scale for the
hierarchy: FP8 local rounding and the subsequent FLOAT32 division can change
both codes and reconstructed values.

.. code-block:: python

    import numpy
    from onnx_light import onnx
    from onnx_light.onnx_core import quantization
    from onnx_light.onnx_py._onnxpykernels import runtime

    def tensor(name, array):
        return runtime.tensor_from_numpy(
            name, onnx.TensorProto.FLOAT, list(array.shape), array.view(numpy.uint8).reshape(-1)
        )

    weights = tensor("W", numpy.ones((2, 32), dtype=numpy.float32))
    local = tensor("scale", numpy.ones((2, 2), dtype=numpy.float32))
    encoded = quantization.quantize_nvfp4_weights(weights, local, 2.0)
    restored = quantization.dequantize_tensor(encoded)
    parameters = quantization.export_nvfp4_weights(encoded)
    encoded_again = quantization.import_nvfp4_weights(parameters)

The ordinary ``EncodedValueProto.SerializeToString`` / ``ParseFromString``
methods serialize the complete value. Its inline descriptor identifies the
version, grouping, matrix shape and FLOAT32 reconstruction; its payload retains
the packed bytes, FP8 local scale bytes and original FLOAT32 global scale.
The raw payload is row-major packed bytes, followed by row-major local-scale
bytes, followed by the four little-endian bytes of the global FLOAT32, with no
padding. Descriptor constants occupy no payload bytes.
``export_nvfp4_weights`` returns independent owned ``weight_packed``
(UINT8 ``[N, K/2]``), ``weight_scale`` (FLOAT8E4M3FN ``[N, K/16]``) and
``weight_global_scale`` (Python float exactly representing the stored FLOAT32).
``import_nvfp4_weights`` validates and serializes these without requantization.
This is lossless parameter export, not a general checkpoint importer.
Both ``dequantize_tensor`` and ``dequantize_tensor_proto`` decode this format.

**Activation-local dynamic scaling is not supported.** Weight support does not
implement or imply parity with the upstream NVFP4 activation preset, its
calibration, training algorithms, or accelerated matrix multiplication.
The portable plan factory and shared-parameter catalogue are not constructors
for this separate fixed weight codec.

Reconstruction precision
------------------------

Precision is a serialized codec contract, **not** the requested output dtype.
For every portable profile (including affine, scalar/vector/additive codebooks,
and cast storage), ``plan.reconstruction_dtype`` selects:

* ``onnx.TensorProto.DOUBLE`` (default): the unchanged
  ``onnx_light.quantization.v1/<profile>`` contract. Block operands and arithmetic
  use binary64. Inverse transforms retain long-double accumulation, rounded to
  binary64 per component; outliers retain their binary64 values.
* ``onnx.TensorProto.FLOAT``: the opt-in
  ``onnx_light.quantization.v2/<profile>`` contract. Parameters remain serialized
  as binary64, but each reconstruction operand is converted to binary32 before
  use. Each operation below rounds separately to binary32, with no fused
  multiply-add:

  * Affine: subtract the zero point from the signed/unsigned code, multiply by
    scale, then add offset.
  * Codebook: start at binary32 zero, add entries in increasing book order, then
    multiply by scale. Each vector component follows the same rule.
  * Cast: convert the stored value to binary32, then multiply by scale.
  * Inverse transform: for each component, start at binary32 zero; in increasing
    row-index order multiply the reconstructed component by the binary32 matrix
    entry, then add the rounded product to the running sum.
  * Scatter the permutation without arithmetic, then restore outliers converted
    to binary32.

The contracts assume IEEE round-to-nearest-even and gradual underflow, not
flush-to-zero or fast-math. Subnormals are retained; underflow may round to zero,
including when converting parameters. Nonfinite serialized parameters, nonfinite
reconstruction operands/results, and overflow raise ``ValueError``. v2 also
rejects finite binary64 operands that overflow during conversion to binary32.
The encoder validates reconstruction using the selected contract.

Only **after** these steps is the result converted to FLOAT16, BFLOAT16, FLOAT
or DOUBLE. FLOAT16/BFLOAT16 conversion uses a FLOAT intermediate and
round-to-nearest-even, unchanged from v1; finite values that overflow the output
dtype are rejected. DOUBLE output for v2 is an exact widening of the FLOAT
reconstruction, not a higher-precision recomputation.

For example, set ``plan.reconstruction_dtype = onnx.TensorProto.FLOAT`` before
calling ``make_quantization_type`` or ``quantize_tensor_proto``. The descriptor
preserves this choice through wire serialization, graph operators, and shared
parameter materialization. Unknown versions and unsupported precision values
are rejected. Existing v1 values are never reinterpreted as v2.

Native decoding remains available to graph/runtime consumers. v2 reconstructs
into ``std::vector<float>``, including its permutation copy and transform row;
it never allocates a full double-valued reconstruction tensor, even for DOUBLE
outputs. Serialized parameter tables, transform matrices and sparse outlier
payloads retain their existing binary64 storage. Encoding/calibration still
uses the original arithmetic. The independent wire goldens in
``unittests/python/backend/test_quantization_precision.py`` express rounding
with explicit NumPy ``float32`` operands/results and check both native consumers.

The three ``ORT_MATMULNBITS_INT*`` profiles retain their v1 DOUBLE decoder and
reject FLOAT reconstruction. This documents the onnx-light decoder, not ORT's
kernel arithmetic. Neither v1 nor v2 claims GGUF numerical compatibility;
GGML/GGUF packing and its FLOAT arithmetic contract are separate codecs.

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

Explicit per-node scales, zero points and offsets accept floating scalar tensors or
one-dimensional tensors with one value per block. They override calibration.
Codebooks concatenate all blocks' tables in run order. FLOAT, DOUBLE, FLOAT16
and BFLOAT16 parameter tensors are supported independently. Permutation and
outlier indices are one-dimensional INT64 tensors. Transform inputs must have
rank two and shape ``[n, n]`` matching the declared transform size, with values
in row-major order. Flattened or reshaped tensors with the same element count
are rejected.

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

.. _l-shared-quantization-parameters:
.. _l-shared-quantization-parameters-python:

Model-level shared parameters
-----------------------------

A storage type describes the arrangement of codes and parameter fields, not
their numerical values. Two nodes using the same ``StructTypeProto.type_ref``
do **not** thereby share scales, zero points, codebooks or transforms. Without
a numerical-parameter reference, each Quantize follows the explicit-input or
automatic-calibration rules above and produces a self-contained payload.

A shared numerical parameter set is a separate model resource. Selecting it
fixes the encoding and reconstruction parameters for every input using that
set; it must not recalibrate scales independently for those inputs. Values
outside the selected range still follow the codec's existing saturation or
nearest-code rules. The actual codes and any saved outlier **values** remain
local to each encoded value.

Declaring and selecting a set
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``add_quantization_parameters(model, name, storage_type, logical_type, *,
scales, ...)`` declares one named set. The optional numerical arguments have
the same names, shapes and dtypes as Quantize's optional inputs. ``scales`` is
required: shared mode never performs automatic calibration. Omitted optional
parameters retain the codec's fixed defaults; learned tables, nonempty
permutations, transforms and outlier indices must still be provided when the
storage descriptor requires them.

The declaration binds the full storage descriptor and a concrete logical tensor
shape and dtype. Every consumer must match them. Quantize selects it with the
string attribute ``parameter_ref`` and still supplies its full ``type``
attribute. Combining a reference with any nonempty optional parameter input
is an error, rather than an override.

The helper returns the **compact output storage type**. Use that type for
encoded graph outputs, not Quantize's full input descriptor. A compact value
contains a reserved byte, local outlier values and packed codes (or ORT ``B``
bytes), and records the set's name in ``EncodedValueProto.parameter_ref``.
Its descriptor describes the actual local byte extent; it does not pretend the omitted parameters are
present in the payload.

For example, two paths can share one scale, including a path through Identity:

.. code-block:: python

    import numpy
    from onnx_light import onnx
    import onnx_light.onnx.numpy_helper as onh
    from onnx_light.onnx import helper
    from onnx_light.onnx_core.quantization import (
        QuantizationFormat,
        add_quantization_parameters,
        make_quantization_plan,
        make_quantization_type,
        materialize_quantized_value,
    )
    from onnx_light.onnx_py._onnxpykernels import runtime

    plan = make_quantization_plan(QuantizationFormat.INT4, 4, block_size=4)
    storage = make_quantization_type(plan)
    destination = onnx.TypeProto(struct_type=storage)
    logical = helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, [4])
    graph = helper.make_graph(
        [], "shared_quantization",
        [
            helper.make_tensor_value_info("X0", onnx.TensorProto.FLOAT, [4]),
            helper.make_tensor_value_info("X1", onnx.TensorProto.FLOAT, [4]),
        ],
        [
            helper.make_tensor_value_info("Y0", onnx.TensorProto.FLOAT, [4]),
            helper.make_tensor_value_info("Y1", onnx.TensorProto.FLOAT, [4]),
        ],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("ai.rt", 1)],
    )
    compact = add_quantization_parameters(
        model, "common", storage, logical, scales=numpy.array(0.5, dtype=numpy.float32),
    )
    for index in range(2):
        model.graph.node.append(helper.make_node(
            "Quantize", [f"X{index}"], [f"Q{index}"], domain="ai.rt",
            type=destination, parameter_ref="common",
        ))
    model.graph.node.append(helper.make_node("Identity", ["Q1"], ["Q1_copy"]))
    for index, encoded_name in enumerate(("Q0", "Q1_copy")):
        model.graph.node.append(helper.make_node(
            "Dequantize", [encoded_name], [f"Y{index}"], domain="ai.rt",
            dtype=onnx.TensorProto.FLOAT,
        ))
    model.graph.output.append(
        helper.make_value_info("Q0", onnx.TypeProto(struct_type=compact)),
    )

    # Model serialization keeps the declaration, initializers and references.
    loaded = onnx.ModelProto()
    loaded.ParseFromString(model.SerializeToString())
    context = runtime.RuntimeContext()
    for name, values in (
        ("X0", [-4, -0.5, 0.5, 3.5]),
        ("X1", [-8, -1, 1, 7]),
    ):
        context.set(name, runtime.tensor_from_proto(
            onh.from_array(numpy.array(values, dtype=numpy.float32), name=name),
        ))
    runtime.RuntimeSession(loaded).run(context)
    numpy.testing.assert_array_equal(numpy.from_dlpack(context.get("Y0")), [-4, -0.5, 0.5, 3.5])
    numpy.testing.assert_array_equal(numpy.from_dlpack(context.get("Y1")), [-4, -1, 1, 3.5])
    shared = context.get_value("Q0")
    assert shared.parameter_ref == "common"
    assert len(shared.raw_data) == 3  # Reserved byte plus four INT4 codes, no scale.
    standalone = materialize_quantized_value(shared)
    assert not standalone.has_parameter_ref()

The second input saturates at the range selected by the shared scale. It is
not independently recalibrated to accommodate its larger values.

Representation, scope and lifetime
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Declarations reuse the root graph's ``quantization_annotation`` catalogue.
An annotation named ``onnx_light.quantization.parameters:<name>`` maps roles
to root ``initializer`` names through ``quant_parameter_tensor_names``.
``storage_type`` and ``logical_type`` are rank-one UINT8 initializers containing
serialized ``StructTypeProto`` and ``TypeProto`` descriptors; the other roles
refer to ordinary numerical TensorProto initializers. Unprefixed annotations
keep their existing meaning. Duplicate set names, duplicate or unknown roles,
missing initializers and incompatible descriptors or numerical tensors are errors.
``checker.check_model`` uses the runtime catalogue's declaration validation and
reports these errors as ``ValidationError``. Numerical parameters stored in
external data must be loaded before this validation.
Catalogue construction derives the fixed plan, shared bytes and local byte
ranges directly from descriptors and supplied parameters. It does not allocate
a logical tensor or a full encoded payload. Its storage depends on parameter
blocks, tables, transforms and indices, not on the number of local code bytes.
This representation uses onnx-light's protobuf model serialization. Native ORT
model serialization rejects quantization annotations and encoded initializers;
textproto export also rejects unsupported structured/shared values instead of
silently dropping their references.

These declarations are model-scoped, not lexical graph inputs. Nested graphs
and model-local functions inherit the model's parameter catalogue; local input
names do not select a different set. Identity preserves the reference and its
owner. Model checking resolves function-bound ``parameter_ref`` attributes using
call-site values or function defaults, including nested calls and subgraphs.
Unresolved attribute references in graph execution scope and omitted required
function attributes are rejected. Shared encoded initializers must match their
catalogue entry's logical type, compact storage type and payload extent;
validation does not reconstruct the full encoded payload.
Session initialization uses the same non-materializing validation before
retaining the compact initializer and its shared parameter catalogue.
The runtime takes an owned, immutable snapshot of the resolved numerical
parameters, so retained encoded outputs can outlive the source model, session
and context. Changing a model after creating its session does not update that
session's parameter snapshot.

Python exposes such retained outputs as ``SharedQuantizedValue`` rather than a
bare proto. ``shared.encoded`` returns an independent snapshot of the compact
encoded message, not a mutable alias of the runtime's value.
``dequantize_tensor(shared)`` and ``materialize_quantized_value(shared)`` use its
retained resource; ``quantize_tensor_shared(tensor, model, name)`` provides the
same fixed-parameter encoding without constructing a graph.

For compact serialization, serialize ``shared.encoded`` and keep the declaring
model alongside it. A parsed bare message has no in-memory owner: pass its model
to the Python dequantization or materialization helper, or place it in that
model's ``extended_initializer`` collection (``encoded_value`` alternative). A reference string alone cannot
recover a missing model resource.

For independent export, call ``materialize_quantized_value(shared)`` first.
It produces a self-contained ``EncodedValueProto`` with inline storage type and
numerical parameters and no ``parameter_ref``. Serialize that result when the
recipient must decode it without the original model. Materialization does not
change the shared source value.

Sharing reduces retained and serialized per-value payloads. The reference codec
still uses temporary full-layout buffers while encoding or materializing a shared
value; this feature does not promise zero-copy quantization or dequantization.

Python
------

The runnable :ref:`Python profile tutorial <l-example-quantization-profiles>`
demonstrates **all 45 profiles**, including per-channel grouping, mixed
precision, supplied vector/additive codebooks, sparse outliers, rotations,
tiling and serialization. The examples use small explicit parameters; they
do not train or calibrate a model.

.. code-block:: python

    import numpy
    import onnx_light.onnx.numpy_helper as onh
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
    encoded = quantize_tensor_proto(onh.from_array(weights), plan)
    restored = onh.to_array(dequantize_tensor_proto(encoded))
    numpy.testing.assert_array_equal(restored, weights)

``plan.runs`` and ``run.blocks`` are converted to/from Python lists of copies.
``plan.run(i)`` returns one run copy; ``plan.set_run(i, run)`` replaces it.
Similarly, ``run.block(j)`` and ``run.set_block(j, block)`` access per-block
parameter copies. No Python object borrows a potentially invalidated vector element.
``quantize_tensor`` and ``dequantize_tensor`` accept/return the native runtime
``Tensor``. Python represents a self-contained encoded ``RuntimeValue`` as
``EncodedValueProto``; shared outputs use the resource-retaining
``SharedQuantizedValue`` described above.

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``quantize_tensor_proto(source, plan)`` returns an owned ``EncodedValueProto``.
``dequantize_tensor_proto(encoded, model=None)`` returns a ``TensorProto``
with the original logical shape and dtype, not the physical code dtype.
Convert it with ``numpy_helper.to_array``. The decoder does not need the
original plan. For self-contained values, scales, tables, transforms and outliers
are encoded with the data; shared values additionally require their model's
numerical parameter set.
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

Numerical contract
------------------

This section describes the portable profiles. The ORT profiles have the
:ref:`matrix-specific contract below <l-ort-matmulnbits-inputs-python>`.

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
    import onnx_light.onnx.numpy_helper as onh
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

    encoded = quantize_tensor_proto(onh.from_array(weights), plan)
    restored = onh.to_array(dequantize_tensor_proto(encoded))
    print(plan.format, numpy.max(numpy.abs(restored - weights)))

All floating-point/codebook profiles use closest-level encoding, with first-entry
ties. They do not promise the external format's float bit patterns or tie-breaking.
Scale tensors and learned codebooks belong to numerical parameter storage, not
shared type constants. They are local to a self-contained value or supplied by
an explicitly selected model parameter set. Consecutive blocks with the same
physical layout share one array element declaration; changing scales or table
contents does not duplicate the descriptor. This reference representation
prioritizes correctness and explicit semantics over minimal payload size or
fast quantization.

Wire layout and validation
--------------------------

This section describes the self-contained portable wire layout. The ORT profiles
use the B/scales/zero_points layout described below instead. Shared values use
the compact layout described in :ref:`l-shared-quantization-parameters-python`.

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
.. _l-ort-matmulnbits-inputs-python:

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
    import onnx_light.onnx.numpy_helper as onh
    from onnx_light.onnx import helper
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
    encoded = quantize_tensor_proto(onh.from_array(weights), plan)
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
