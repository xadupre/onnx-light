.. _l-gptqmodel-compatibility:

GPTQModel checkpoint compatibility
=================================

This is the format-level tracker for :issue:`5792`, alongside
:ref:`compressed-tensors support <l-compressed-tensors-checkpoints>`.
It inventories **one module's loaded weight tensors**, not calibration,
training algorithms, accelerated inference or whole-model loading.
Numerical profiles are not checkpoint ABIs: existing portable onnx-light
payloads and decoders are not substitutes for any layout listed here.
The detailed guides linked below define the accepted geometry, dtypes,
parameters and reconstruction arithmetic; a family name alone is insufficient.

Revision and container boundaries
--------------------------------

The GPTQModel baseline for every row is
``c1976918cfe3703fbcf6d1ce2440a82b26394375``:
`configuration <https://github.com/ModelCloud/GPTQModel/blob/c1976918cfe3703fbcf6d1ce2440a82b26394375/gptqmodel/quantization/config.py>`_
and `packing guide <https://github.com/ModelCloud/GPTQModel/wiki/Packing-Formats-and-Compatibility>`_.
The wiki describes that baseline; it is not a versioned implementation pin
or a claim about current main. Each detailed guide identifies implementation
sources and any additional dependency revisions. In particular:

* GGUF blocks pin llama.cpp/GGML
  ``0c1e57098bba43ac29e6e3b677cdceebdd22334f`` (gguf-py 0.19.0).
* bitsandbytes state pins 0.50.2,
  ``08a9956b939857ae2c4fbd56753960cc370fa7cf``.
* EXL3 and VPTQ use GPTQModel's vendored implementations, not an arbitrary
  installed ExLlamaV3 or Microsoft VPTQ package.
* Target-specific BitBLAS transforms have no supported external ABI pin here.
* The separate compressed-tensors baseline is
  ``eb058d630f7344099459b6795b1e3eeffdda4b40``.

Minimum dependency versions upstream do not freeze an ABI. Verify the actual
producer, decoder and dependency revisions before extending any support claim.
Unknown revisions, schemas and layouts must fail clearly, never select a
portable profile silently.

All supported imports below take already-loaded tensors with explicit resolved
metadata. SafeTensors payload adaptation is **not** SafeTensors file/shard
discovery or configuration parsing. Native ``.gguf`` container import/export
is **unsupported**: exact GGML block support does not supply a container
header/directory parser. No generic checkpoint container framework is required.

Import/export support matrix
----------------------------

``Yes`` means support within the linked contract's validated subset at the
baseline above, not every upstream release or optional backend. Unless noted,
the adapter schema is onnx-light v1, not an upstream file-format version.
Export preserves imported packed codes, parameter bits, padding and supported
auxiliary state; it does not quantize dense weights.

.. list-table::
   :header-rows: 1
   :widths: 18 42 10 30

   * - Family / tracking issue
     - Explicit revision/subtype contract
     - Import
     - Export / limitations
   * - :ref:`GPTQ <l-gptq-checkpoints>`; :issue:`5784`
     - ``gptq`` (v1), ``gptq_v2``: 2/4/8-bit continuous words,
       3-bit INT32 streams; ``gptq_p``: 2--8-bit INT32 planes.
       Retains zero convention and ``g_idx``.
     - Yes
     - Yes, same layout; v1 ambiguous zero extrema are rejected.
   * - :ref:`AWQ <l-awq-checkpoints>`; :issue:`5785`
     - ``gemm`` (4/8-bit), ``gemv``, ``gemv_fast``, ``llm-awq``
       (4-bit); distinct packing and zero/offset conventions.
     - Yes
     - Yes, same layout, including rounded scaled-zero parameters.
   * - :ref:`GGUF-family <l-gguf-blocks>`; :issue:`5783`
     - ``ggml-blocks-v1``: Q4_0, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K,
       Q6_K, IQ4_NL, IQ1_S, TQ1_0, TQ2_0.
     - Yes
     - Yes, exact blocks/extracted UINT8 payloads; other subtypes and
       native ``.gguf`` containers unsupported.
   * - :ref:`bitsandbytes <l-bitsandbytes-checkpoints>`; :issue:`5786`
     - NF4/FP4, including nested scale state; INT8 plus SCB.
       GPTQModel module-local ``weight_*`` schema, not dotted HF keys.
     - Yes
     - Yes, including QuantState JSON bytes; not arbitrary CPU repacks,
       outlier-routing or optimizer state.
   * - :ref:`EXL3 <l-exl3-checkpoints>`; :issue:`5782`
     - Integer trellis rates 1--8; ``3inst``, ``mcg``, ``mul1``;
       folded ``suh``/``svh`` scales and specified Hadamard transforms.
     - Yes
     - Yes; custom codebooks, legacy separate sign/transform state and
       fractional-rate bitstreams unsupported.
   * - :ref:`VPTQ <l-vptq-checkpoints>`; :issue:`5790`
     - Packed main/residual indices, codebooks, optional outliers,
       permutation, normalization and bias; FP16/BF16/FP32 state.
     - Yes
     - Yes; not external Microsoft VPTQ or unpacked main-index layouts.
   * - :ref:`QQQ <l-qqq-checkpoints>`; :issue:`5788`
     - Four-bit tiled ``B`` plus ``s_channel``/``s_group``;
       group size -1 or 128, with distinct channel/group decoding.
     - Yes
     - Yes; weight reference includes grouped rounding/saturation,
       not dynamic INT8 activation quantization.
   * - :ref:`ParoQuant <l-paroquant-checkpoints>`; :issue:`5791`
     - Four-bit AWQ GEMM plus ``pairs``, ``theta``, ``channel_scales``
       and optional bias; retains rotation/operator semantics.
     - Yes
     - Yes; reconstructed weights alone omit the activation transform.
       Generic weight dequantization rejects this operator schema.
   * - :ref:`Historical Marlin <l-marlin-bitblas-checkpoints>`; :issue:`5789`
     - Explicit ``gptq-v1`` or ``awq-gemm`` source layout, 4/8-bit;
       uses ordinary GPTQ/AWQ encoded schemas.
     - Yes
     - Through ordinary GPTQ/AWQ export only; no Marlin tile exporter.
       Runtime tiles and historical ``B/s`` buffers unsupported.
   * - :ref:`BitBLAS <l-marlin-bitblas-checkpoints>`; :issue:`5789`
     - ``row-packed-v1`` from the pinned CPU row reader:
       GPTQ 2/4/8-bit; AWQ 2--8-bit; explicit originating method.
     - Yes
     - Yes, row-packed only; target-specific ladder/LOP3/fast-decoding
       transforms unsupported, regardless of equal buffer shapes.
   * - :ref:`Native FP8 <l-gptqmodel-fp8-checkpoints>`; :issue:`5787`
     - ``float8_e4m3fn``, ``float8_e5m2``, ``float8_e4m3fnuz``,
       ``float8_e5m2fnuz``, ``float8_e8m0fnu``; tensor/row/block
       FLOAT32 inverse scales.
     - Yes
     - Yes, original inverse-scale bits; E8M0 weight subtype is
       reconstruction-only upstream, not an MXFP8 scale schema.

Lossless export means supported tensor state plus resolved metadata, not
byte-for-byte reproduction of a source file or unrecognized side tensors.
Dense reconstruction is a numerical reference, not a reversible checkpoint
conversion or a promise of bit-identical accelerator arithmetic.
Cross-layout conversion requires explicit unpack/repack and representable
parameters: for example, GPTQ v2 zero 0 cannot be represented in validated v1.
FP8 inverse-to-direct scales and dense requantization can be lossy. Neither is
performed implicitly.

Reuse and independent work
-------------------------

* :issue:`5781` owns compressed-tensors adapters and common loaded-array,
  metadata, fixture and low-bit facilities. Its ``pack-quantized``
  offset-binary INT32 ABI is **not GPTQ/AWQ packing**.
* :issue:`5780` owns MXFP4 E2M1/group-32/E8M0;
  :issue:`5777` owns MXFP6 E2M3 while preserving portable E3M2 payloads;
  :issue:`5779` owns MXFP8 E4M3FN/group-32/E8M0;
  :issue:`5778` owns NVFP4 hierarchical scales. Their
  :ref:`existing contracts <l-mxfp4-weights>` are reused for equivalent
  variants; GPTQModel-specific work adds only verified schema/layout handling.
  These numerical contracts do not by themselves establish every GPTQModel
  MX/NV checkpoint schema.
* Ordinary FP8 with inverse scales is **not MXFP8** and does not depend on
  :issue:`5779`. Actual MX/NV variants depend on the corresponding contracts,
  not merely a matching dtype or format label.
* ParoQuant reuses the exact :issue:`5785` AWQ primitives and retains its
  additional state/operator semantics. Marlin/BitBLAS reuse applicable
  :issue:`5784`/:issue:`5785` primitives, without inventing a transformed ABI
  from an inference-backend name.
* Other families proceed independently. Shared utilities should be reused
  where applicable, not made a universal framework or a blanket dependency
  on completion of :issue:`5781`.

Evidence and completion gates
-----------------------------

The existing CPU-only evidence is indexed below. Paths are repository-relative.
Fixtures may be checked-in JSON or source-derived packed constants/reference
builders inside tests; a round trip through the same codec alone is not a
reference oracle. The linked family guides record provenance and reconstruction
precision. Torch/upstream packages may be needed to regenerate some fixtures,
not to run the native tests or use the core adapters.

.. list-table::
   :header-rows: 1

   * - Family
     - Native test / pinned fixture evidence
   * - GPTQ
     - ``unittests/cc/onnx_core/runtime/test_gptq.cc``
   * - AWQ
     - ``unittests/cc/onnx_core/runtime/test_awq.cc``
   * - GGUF-family
     - ``unittests/cc/onnx_core/runtime/test_gguf.cc``;
       ``unittests/python/backend/data/gguf_native_v1.json``
   * - bitsandbytes
     - ``unittests/cc/onnx_core/runtime/test_bitsandbytes.cc``;
       ``unittests/python/backend/data/bitsandbytes_native_v1.json``
   * - EXL3
     - ``unittests/cc/onnx_core/runtime/test_exl3.cc``;
       ``unittests/python/backend/test_exl3.py`` (reference reconstruction)
   * - VPTQ
     - ``unittests/cc/onnx_core/runtime/test_vptq.cc``;
       ``unittests/python/backend/data/vptq_native_v1.json``
   * - QQQ
     - ``unittests/cc/onnx_core/runtime/test_qqq.cc``;
       ``unittests/python/backend/data/qqq_native_v1.json``
   * - ParoQuant
     - ``unittests/cc/onnx_core/runtime/test_paroquant.cc``
   * - Historical Marlin
     - ``unittests/python/backend/test_marlin.py`` plus the reused native
       GPTQ/AWQ tests above.
   * - BitBLAS
     - ``unittests/cc/onnx_core/runtime/test_bitblas.cc``;
       ``unittests/python/backend/data/bitblas_native_v1.json``
   * - Native FP8
     - ``unittests/cc/onnx_core/runtime/test_gptqmodel_fp8.cc``

For each new revision/subtype, completion requires all of:

* Pinned implementation/dependency provenance and lightweight independent
  reference fixtures, with exact packed-code and parameter-bit assertions.
* Reference reconstruction covering orientation, grouping, zero conventions
  and every required auxiliary transform/state tensor.
* Malformed schema/metadata/tensor tests, truncated and oversized payload
  rejection, checked extent/product overflow and pre-allocation resource limits.
  Fast paths, identity transforms and zero/default parameters must not bypass
  validation.
* Explicit import/export and lossless/lossy boundaries, container versus
  tensor-layout scope, and rejection tests for unsupported versions/layouts.
* Native tests without a GPU or unconditional PyTorch dependency in the core,
  and regression coverage preserving unrelated portable payload/decoder ABIs.

The matrix deliberately retains unsupported boundaries rather than marking
entire format families universally complete.
