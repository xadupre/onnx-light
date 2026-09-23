.. _l-quantized-values:

Quantizes tensors into encoded values
====================================

The converters implement **portable onnx-light representations** of the
:ref:`quantization catalogue <l-next-steps-quantization>`. They do not emit
GGUF, Marlin, bitsandbytes or other vendor-compatible buffers. The profile
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

Python
------

.. code-block:: python

    import numpy
    from onnx_light.onnx import numpy_helper
    from onnx_light.onnx_core.quantization import (
        make_quantization_plan,
        quantize_tensor_proto,
        dequantize_tensor_proto,
    )

    weights = numpy.array([[-4, 0, 3.5], [-32, 0, 30]], dtype=numpy.float32)
    plan = make_quantization_plan("exl2", weights.size, block_size=3)
    blocks = plan.blocks
    blocks[0].bits = 4
    blocks[0].scale = 0.5
    blocks[1].bits = 5
    blocks[1].scale = 2
    plan.blocks = blocks
    encoded = quantize_tensor_proto(numpy_helper.from_array(weights), plan)
    restored = numpy_helper.to_array(dequantize_tensor_proto(encoded))
    numpy.testing.assert_array_equal(restored, weights)

``blocks`` is converted to/from a Python list of copies. ``block(i)`` returns
one copy; ``set_block(i, block)`` replaces it. No Python object borrows a
potentially invalidated element of a C++ vector.
``quantize_tensor`` and ``dequantize_tensor`` accept/return the native runtime
``Tensor``. As with runtime structured feeds, Python represents an encoded
``RuntimeValue`` as ``EncodedValueProto`` rather than introducing another wrapper.

C++
---

.. code-block:: cpp

    #include "onnx_core/runtime/quantization.h"

    using namespace onnx_light::core::runtime;
    Tensor input = Tensor::FromFloat("weight", {3}, {-4, 0, 3.5f});
    auto plan = MakeQuantizationPlan("int4", 3);
    plan.blocks[0].scale = 0.5;
    RuntimeValue encoded = QuantizeTensor(input, plan);
    Tensor restored = DequantizeTensor(encoded);

``QuantizeTensorProto`` and ``DequantizeTensorProto`` provide the corresponding
message conversions. C++ dequantizers accept a ``StructTypeCatalogue`` for
model-scoped references; Python dequantizers accept an optional ``model``.
The C++ tensor dequantizer also accepts an allocator.

Numerical contract
------------------

Inputs and logical outputs support FLOAT, DOUBLE, FLOAT16 and BFLOAT16.
Input shapes are concrete, including scalars and empty tensors. Inputs must
be finite. Source storage is never modified and results own their storage.
Nonfinite reconstructions and floating-point cast overflows are rejected.
External messages must have their payload loaded before conversion.

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

Scales default to **one**, not to an automatically estimated calibration.
Callers supply their scales, integer zero points, real offsets, trained
codebooks, rotations and selected outlier indices. Missing learned tables
and required rotations raise an error. No GPTQ Hessian calculation, AWQ
calibration, QAT or codebook training is performed.

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

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Profiles
     - Portable representation and caller inputs
   * - ``int8``, ``int8_per_channel``, ``eetq``, ``int4``
     - Signed affine codes; scales and channel grouping supplied.
   * - ``gptq``, ``awq``, ``matmulnbits``
     - Unsigned INT4 affine codes, default zero point 8; supplied group parameters.
   * - ``q2_k``, ``q3_k``, ``q4_k``, ``q5_k``, ``q6_k``
     - 2--6-bit affine blocks; supplied effective sub-block scales and offsets.
   * - ``hqq``, ``exl2``, ``exl3``
     - Independently configurable affine blocks; override bits, counts and parameters
       for mixed precision, or select a codebook block explicitly.
   * - ``nf4``, ``iq4_nl``
     - Fixed scalar codebooks and supplied scales. NF4 uses the full-precision
       normal-float table; IQ4_NL uses its 16 signed integer levels.
   * - ``binary``
     - One-bit indices into ``[-1, 1]``.
   * - ``ternary``, ``tq1_0``, ``bitnet``, ``paretoq``, ``tequila``
     - Indices into ``[-1, 0, 1]``; five base-3 digits per byte.
       This represents ternary weights, not the associated training procedures.
   * - ``tq2_0``
     - The same ternary table with two bits per index.
   * - ``stq1_0``
     - Supplied vector codebook; defaults to 32 four-component entries and 5-bit
       indices. Code/sign splitting and vendor scatter layouts are not emitted.
   * - ``iq1_s``, ``quip_sharp``
     - Supplied vector codebooks; defaults to 256 eight-component entries.
       QuIP# also requires an explicit transform pair.
   * - ``aqlm``
     - Supplied additive vector codebooks; defaults to two 256-entry,
       eight-component books with 8-bit indices.
   * - ``spqr``, ``squeezellm``
     - Affine or supplied scalar-codebook base, respectively, plus exact sparse
       outliers. SqueezeLLM defaults to 16 supplied levels. Outlier selection is
       the caller's responsibility.
   * - ``log``
     - Scalar codebook with zero and signed powers of two from ``2**-3`` through
       ``2**3``. Supply another table for a different base, range or logarithmic rule.
   * - ``fp6_llm``, ``mxfp6``
     - E3M2 finite levels and supplied scales, using 6-bit codebook indices.
   * - ``mxfp4``, ``nvfp4``
     - E2M1 finite levels and supplied effective scales. The caller rounds scales
       to E8M0/FP8 and combines scale levels if required by their numerical profile.
   * - ``fp8_e4m3``
     - Finite E4M3FN levels with 8-bit codebook indices; nonfinite levels excluded.
   * - ``quarot``, ``smoothquant``
     - Affine blocks with explicit forward/inverse rotations or rescaling.
   * - ``tiled_float``, ``column_major``
     - FLOAT casts plus an explicitly supplied ordering permutation.

All floating-point/codebook profiles use closest-level encoding, with first-entry
ties. They do not promise the external format's float bit patterns or tie-breaking.
Scale tensors and learned codebooks belong to payload storage, not shared type
constants. Consecutive blocks with the same physical layout share one array
element declaration; changing scales or table contents does not duplicate the
descriptor. This reference representation prioritizes correctness and explicit
semantics over minimal payload size or fast quantization.

Wire layout and validation
--------------------------

The structured root name is ``onnx_light.quantization.v1/<profile>``. Its fields
are, in order: one zero reserved byte, an INT64 permutation, DOUBLE forward and
inverse matrices, INT64 outlier indices, DOUBLE outlier values, and a structure
of block runs with a constant total block count. Each run is an array of blocks
with identical layout parameters, named ``run_<first-block-index>``. Adjacent
compatible runs are merged. Integers/floats in the payload are little-endian.

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
