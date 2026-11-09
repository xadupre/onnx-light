.. _l-gguf-blocks:

Exact GGUF-family block payloads
===============================

``onnx_light.onnx_core.gguf`` and ``onnx_core/runtime/gguf.h`` store exact
little-endian GGML weight blocks in the standard ``EncodedValueProto``.
Its ``raw_data`` contains the original blocks without an onnx-light packing
layer; its logical type records the FLOAT32 tensor shape. The versioned
structured type records the canonical subtype and pinned upstream revision.
The usual ``dequantize_tensor`` and ``dequantize_tensor_proto`` APIs decode
these values using the native GGML codec.

All supported formats belong to the single ``QuantizationFormat`` catalogue:
``Q4_0``, ``Q8_0``, ``Q2_K``, ``Q3_K``, ``Q4_K``, ``Q5_K``, ``Q6_K``,
``IQ4_NL``, ``IQ1_S``, ``TQ1_0`` and ``TQ2_0`` appear in
``quantization_formats()`` and use their canonical uppercase names in
``quantization_format_name()`` and ``parse_quantization_format()``.
``gguf_block_layouts()`` provides their physical block sizes, not a separate
format catalogue. Native formats accept existing blocks; ``make_quantization_plan``
rejects them because floating-point-to-GGML quantization is not implemented.

The former generic ``q2_k``/``q3_k``/``q4_k``/``q5_k``/``q6_k``,
``iq4_nl``, ``iq1_s``, ``tq1_0`` and ``tq2_0`` codecs are replaced by their
native GGML implementations in that same catalogue, not removed as formats.
Their old v1/v2 encoded values are rejected, not migrated or reinterpreted.
Codecs without a GGUF counterpart retain their existing representations.
Unsupported GGML subtypes are rejected rather
than replaced with generic affine or codebook packing. Generic numerical
experiments can still customize an ``int4`` or ``ternary`` layout, without
claiming a GGML format name.

Version identity
----------------

This stage defines ``LAYOUT_VERSION = "ggml-blocks-v1"``. Its checkpoint
inventory is pinned to GPTQModel
``c1976918cfe3703fbcf6d1ce2440a82b26394375`` (``UPSTREAM_REVISION``).
The native fixture/reference pin is GGML in llama.cpp
``0c1e57098bba43ac29e6e3b677cdceebdd22334f`` (``REFERENCE_REVISION``),
the gitlink shipped by llama-cpp-python v0.3.36, with gguf-py 0.19.0.
GPTQModel's optional dependency constraints are minimum versions, so its
revision alone does not pin a particular installed native quantizer.
The layout version is **not** a GGUF container version.

Sources:

* `GPTQModel subtype tables and CPU reconstruction
  <https://github.com/ModelCloud/GPTQModel/blob/c1976918cfe3703fbcf6d1ce2440a82b26394375/gptqmodel/nn_modules/qlinear/gguf.py>`_
* `GPTQModel quantization metadata
  <https://github.com/ModelCloud/GPTQModel/blob/c1976918cfe3703fbcf6d1ce2440a82b26394375/gptqmodel/quantization/config.py>`_
* `Native block structures
  <https://github.com/ggml-org/llama.cpp/blob/0c1e57098bba43ac29e6e3b677cdceebdd22334f/ggml/src/ggml-common.h>`_
  and `reference codecs
  <https://github.com/ggml-org/llama.cpp/blob/0c1e57098bba43ac29e6e3b677cdceebdd22334f/ggml/src/ggml-quants.c>`_
* `Independent Python reconstruction
  <https://github.com/ggml-org/llama.cpp/blob/0c1e57098bba43ac29e6e3b677cdceebdd22334f/gguf-py/gguf/quants.py>`_

Staged support matrix
---------------------

The pinned GPTQModel tables recognize these quantized subtypes. Recognition
upstream does not imply availability of every packing backend: some depend
on optional native libraries. The following table inventories all entries,
including the sign-only compatibility alias.

.. list-table::
   :header-rows: 1
   :widths: 45 15 15 25

   * - Canonical tensor subtype
     - Values/block
     - Bytes/block
     - onnx-light stage
   * - Q4_0
     - 32
     - 18
     - Supported
   * - Q8_0
     - 32
     - 34
     - Supported
   * - Q4_K / Q5_K / Q6_K
     - 256
     - 144 / 176 / 210
     - Supported
   * - Q1_0 / Q1_0_g128
     - 128
     - 18
     - Deferred; rejected
   * - Q2_0
     - 64
     - 18
     - Deferred; rejected
   * - Q4_1 / Q5_0 / Q5_1 / Q8_1
     - 32
     - 20 / 22 / 24 / 36
     - Deferred; rejected
   * - Q2_K / Q3_K
     - 256
     - 84 / 110
     - Supported
   * - Q8_K
     - 256
     - 292
     - Deferred; rejected
   * - IQ1_S
     - 256
     - 50
     - Supported
   * - IQ1_M
     - 256
     - 56
     - Deferred; rejected
   * - IQ2_XXS / IQ2_XS / IQ2_S
     - 256
     - 66 / 74 / 82
     - Deferred; rejected
   * - IQ3_XXS / IQ3_S
     - 256
     - 98 / 110
     - Deferred; rejected
   * - IQ4_NL
     - 32
     - 18
     - Supported
   * - IQ4_XS
     - 256
     - 136
     - Deferred; rejected
   * - TQ1_0 / TQ2_0
     - 256
     - 54 / 66
     - Supported
   * - MXFP4 / NVFP4
     - 32 / 64
     - 17 / 36
     - Excluded here; rejected

MXFP/NVFP numerical and checkpoint APIs are separate work; this stage does
not reuse their codecs for GGUF layouts. Raw storage types
F32/F16/BF16/I8/I16/I32/I64/F64 and architecture-specific repacks are not
quantized targets in this matrix and are rejected too.

Only exact uppercase names returned by ``gguf_block_layouts()`` are accepted.
No inference from bit width, byte count or ``"gguf"`` is performed.
GPTQModel's aliases ``q2_k_s/m``, ``q3_k_s/m/l``, ``q4_k_s/m`` and
``q5_k_s/m`` select the corresponding base K block layout upstream, not
different physical codecs. They are not accepted as tensor subtype names
by this API. Unknown subtypes, versions and revisions always raise errors.
Future stages require their own native fixtures before claiming support.

Support boundaries
------------------

.. list-table::
   :header-rows: 1

   * - Layer
     - This stage
   * - Block codec
     - Exact validation, FLOAT32 reconstruction, field inspection and
       byte-preserving import/export for all eleven supported subtypes.
   * - GPTQModel SafeTensors adaptation
     - Extracted UINT8 ``qweight`` matrices with explicit canonical subtype,
       logical shape and pinned metadata. No whole-checkpoint/shard discovery,
       configuration parsing, bias loading or automatic tensor-name mapping.
   * - Standalone ``.gguf`` container
     - Not implemented: no header/directory parser, model loader or exporter.
   * - Quantization from floating-point weights
     - Not implemented: export preserves imported codes, not calibration or
       upstream quantizer choices. GPTQModel fallback and native quantizers
       can differ at rounding boundaries.

For a checkpoint linear layer, the logical shape is ``[out_features,
in_features]``; ``qweight`` is ``uint8[out_features,
in_features / block_size * type_size]``. This API reconstructs the original
``[out,in]`` orientation, not GPTQModel's transposed ``dequantize_weight()``
result. Scales/minima reside inside each block; GPTQ ``scales``, ``qzeros``
and ``g_idx`` are not substitutes.

Checkpoint configuration must be resolved externally and unambiguously:
``method=gguf, bits=4, format=q_0`` selects Q4_0, while width 8 with the
same public format selects Q8_0. Likewise ``q_k`` needs the width to select
Q4_K/Q5_K/Q6_K. A generic format string alone is insufficient.

Block layout and precision
--------------------------

All offsets below are bytes within a block. Binary16 fields are decoded to
FLOAT32 before arithmetic. Codes and subgroup fields are also promoted to
FLOAT32. Subgroup scale multiplication, code multiplication and minimum
subtraction use separate FLOAT32 operations, never a portable double-affine
profile. Results match the pinned native/NumPy FLOAT32 reconstruction.
GPTQModel's direct lower-precision GPU reconstruction is not the oracle.

* **Q4_0:** binary16 ``d`` at 0; sixteen code bytes at 2. Low nibbles
  reconstruct elements 0--15; high nibbles reconstruct 16--31. Subtract 8
  from the nibble code, then multiply by ``d``. Negative scales are valid.
* **Q8_0:** binary16 ``d`` at 0; thirty-two signed two's-complement
  int8 codes at 2. Reconstruction is ``d * code``.
* **Q2_K:** sixteen scale/minimum bytes at 0, sixty-four code bytes at 16,
  and binary16 ``d``/``dmin`` at 80/82. Each 16-value subgroup uses the
  low/high scale nibble. Two-bit code planes reconstruct
  ``(d * scale) * code - dmin * minimum`` with separate FLOAT32 rounding.
* **Q3_K:** thirty-two high-mask bytes at 0, sixty-four low-code bytes at 32,
  twelve packed six-bit scale bytes at 96, and binary16 ``d`` at 108.
  Scales are centered at 32; the high mask selects whether to subtract four
  from the low two-bit code.
* **IQ4_NL:** binary16 ``d`` at 0 and sixteen paired nibble bytes at 2.
  Low nibbles reconstruct the first sixteen values, high nibbles the next
  sixteen, through GGML's fixed signed integer table.
* **IQ1_S:** binary16 ``d`` at 0, thirty-two low grid-index bytes at 2,
  and eight little-endian subgroup words at 34. Each word contains four
  three-bit index extensions, a three-bit odd scale multiplier and the sign
  of a dimensionless 0.125 offset. The exact 2048-by-eight ternary grid
  comes from the pinned upstream reference.
* **TQ1_0:** forty-eight scaled base-three bytes, four high bytes, and
  binary16 ``d`` at 52. Byte multiplication wraps modulo 256 before
  multiplying by three and extracting its high byte; 32/16/4-lane planes
  preserve native ordering.
* **TQ2_0:** sixty-four two-bit code bytes and binary16 ``d`` at 64.
  Four 32-lane planes per 128 values reconstruct ``d * (code - 1)``.
* **Q4_K/Q5_K:** binary16 ``d`` and ``dmin`` at 0 and 2; twelve packed
  scale/minimum bytes at 4. These reconstruct eight unsigned 6-bit scale
  and minimum pairs. The first four pairs use low six bits; high two bits
  in the first eight bytes supply the upper bits of the last four pairs,
  whose low four bits share bytes 8--11.
  Q4_K has 128 nibble bytes at 16. Q5_K has 32 high-bit bytes at 16
  and 128 nibble bytes at 48. Within each 64-value segment, low and high
  nibbles represent its first and second 32-value groups. Q5_K high-bit
  byte ``j`` supplies bit 4 for lane ``j`` in each of eight groups.
  Reconstruction is ``((d * scale[group]) * code) -
  (dmin * minimum[group])``.
* **Q6_K:** low nibbles at 0 (128 bytes), upper two-bit planes at 128
  (64 bytes), sixteen signed int8 scales at 192, binary16 ``d`` at 208.
  Each 128-value half combines four 32-value planes. Low nibbles from
  the first/second 32-byte spans feed planes 0/1, high nibbles feed
  planes 2/3; high-bit bytes supply two bits per plane.
  Subtract 32 from the combined six-bit code; select the signed subgroup
  scale by logical element index divided by 16.
  Reconstruction is ``(d * scale[group]) * code``.

``read_gguf_block_parameters()`` returns effective FLOAT32 scales (``d``
times subgroup scale) and subtractive minima (``dmin`` times subgroup
minimum), in logical subgroup order. Symmetric formats return no minima.
It validates metadata and the requested block only, without scanning other blocks.
Export retains the original binary16 fields and packed scale/minimum bytes.

Validation and partial blocks
-----------------------------

Logical shapes must have rank at least one, with all dimensions positive.
The last dimension must be divisible by the subtype's block size; flat
element-count divisibility alone is insufficient. Scalar, empty and partial
blocks are rejected, as are truncated or trailing bytes.
GPTQModel can zero-pad its original input dimension to a block boundary;
**this stage rejects those partial original dimensions**, rather than
guessing a shape, trimming values or changing stored bytes.

NaN and infinity in any binary16 parameter are rejected, even when a zero
subgroup multiplier would hide them. Finite negative values, signed zero
and binary16 subnormals are accepted. Every reconstructed value must be
finite. Consumers revalidate mutable payload metadata and the bytes they read;
import and export validate every block.
Checked products reject dimension/byte-size overflow before allocation.
``max_bytes`` (default 1 GiB) separately bounds both packed bytes and
potential FLOAT32 reconstruction bytes, including on import and export;
it is not a total process-memory budget.

Usage
-----

.. code-block:: python

    import numpy
    from onnx_light.onnx_core import gguf

    # qweight is an extracted SafeTensors UINT8 array; do not reinterpret INT32.
    qweight = numpy.zeros((1, 18), dtype=numpy.uint8)
    checkpoint = gguf.GptqModelGgufWeights(qweight, [1, 32], "Q4_0")
    encoded = gguf.import_gptqmodel_gguf_weights(checkpoint)
    tensor = gguf.runtime.dequantize_tensor(encoded)
    values = gguf.runtime.tensor_to_numpy(tensor).view(numpy.float32).reshape(1, 32)
    exported = gguf.export_gptqmodel_gguf_weights(encoded)
    assert exported.qweight.tobytes() == qweight.tobytes()
    raw = gguf.export_gguf_tensor_payload(encoded)
    restored = gguf.import_gguf_tensor_payload(raw, [1, 32], "Q4_0")

The C++ counterparts are ``ImportGgufTensorPayload``,
``ExportGgufTensorPayload``, ``DequantizeGgufTensorPayload``,
``ReadGgufBlockParameters``, ``ImportGptqModelGgufWeights`` and
``ExportGptqModelGgufWeights``.

Native fixture provenance
-------------------------

``unittests/python/backend/data/gguf_native_v1.json`` contains little-endian
native blocks and independent FLOAT32 reconstruction goldens for every
supported subtype, plus effective scale/minimum fields.
The original five fixtures use ``quantize_row_*_ref`` bytes, bitwise
cross-checked between native ``dequantize_row_*`` and pinned gguf-py.
The six additional fixtures use deterministic packed blocks decoded through
the pinned gguf-py implementation, including extreme finite parameters.
Each regular fixture contains two rows with two blocks per row. An additional
IQ1_S fixture exercises all 2048 grid entries and subgroup scale/offset fields.
Both source revisions and producer identities are recorded in the file.
Subgroup fields were unpacked independently, not through this implementation.
Zlib/base64 is solely
lossless fixture storage, not a weight block format or runtime dependency.
Tests require neither GPTQModel nor GGML to be installed.
