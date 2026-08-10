
.. _l-next-steps-quantization:

Quantization
============

:Date: 2026-08

**complete**

.. note::

    The structures on this page are descriptive quantization profiles. They
    should be implemented with :ref:`l-next-steps-custom-types`, not as a
    parallel protobuf hierarchy. Each specialized structure below is followed
    by its ``StructTypeProto`` translation. The families remain useful
    as format names, validation profiles, and decoder specifications.

Format coverage summary
+++++++++++++++++++++++

.. list-table::
   :header-rows: 1
   :widths: 25 8 25 10

   * - Format
     - bpw
     - Proto used
     - Possible
   * - 1.58-bit Ternary (BitNet)
     - 1.63
     - ``Codebook``
     - ✅
   * - AQLM 2×8
     - 3.0
     - ``Codebook``
     - ✅
   * - Binary/XNOR (1-bit)
     - 1.0
     - ``Codebook``
     - ✅
   * - EXL2 (variable bpw)
     - 3.5
     - ``Tiling`` + ``Linear`` (multiple)
     - ✅
   * - EETQ (INT8 weight-only)
     - 8.0
     - ``Linear``
     - ✅
   * - EXL3 (improved EXL2)
     - 2–6
     - ``Tiling`` + ``Linear`` or ``Codebook``
     - ✅
   * - FP6 LLM (TC-FPn)
     - 6.125
     - ``Tiling`` + ``FloatingPoint``
     - ✅
   * - FP8 E4M3
     - 8.0
     - ``FloatingPoint``
     - ✅
   * - HQQ (mixed-precision per head)
     - 2–4
     - ``Tiling`` + ``Linear``
     - ✅
   * - INT4 AWQ (per-group)
     - 4.5
     - ``Tiling`` + ``Linear``
     - ✅
   * - INT4 GPTQ (per-group)
     - 4.5
     - ``Tiling`` + ``Linear``
     - ✅
   * - INT4 Symmetric
     - 4.5
     - ``Linear``
     - ✅
   * - INT8 per-channel
     - 8.0
     - ``Linear``
     - ✅
   * - INT8 symmetric
     - 8.0
     - ``Linear``
     - ✅
   * - IQ1_S
     - 1.56
     - ``Tiling`` + ``Codebook``
     - ✅
   * - IQ4_NL
     - 4.5
     - ``Codebook``
     - ✅
   * - Log quantization
     - 4.0
     - ``Log``
     - ✅
   * - MatMulNBits INT4 (ORT)
     - 4.5
     - ``Tiling`` + ``Linear``
     - ✅
   * - MXFP4
     - 5.0
     - ``Tiling`` + ``FloatingPoint``
     - ✅
   * - MXFP6 E3M2
     - 6.125
     - ``Tiling`` + ``FloatingPoint``
     - ✅
   * - NF4 (QLoRA)
     - 4.5
     - ``Codebook``
     - ✅
   * - NVFP4 (E2M1 + FP8 scale)
     - 4.5
     - ``Tiling`` + ``FloatingPoint``
     - ✅
   * - Q2_K
     - 2.625
     - ``Tiling`` × 2 + ``Linear``
     - ✅
   * - Q3_K
     - 3.4
     - ``Tiling`` × 2 + ``Linear``
     - ✅
   * - Q4_K
     - 4.5
     - ``Tiling`` × 2 + ``Linear``
     - ✅
   * - Q5_K
     - 5.5
     - ``Tiling`` × 2 + ``Linear``
     - ✅
   * - Q6_K
     - 6.6
     - ``Tiling`` × 2 + ``Linear``
     - ✅
   * - QuaRot (rotational quantization)
     - 4.0
     - ``Linear`` + ``RotationProto``
     - ✅
   * - QuIP#
     - 2.0
     - ``Codebook`` + rotation
     - ✅
   * - SpQR
     - 3.4
     - ``Sparse`` + ``Tiling`` + ``Linear``
     - ✅
   * - SmoothQuant (W8A8)
     - 8.0
     - ``Linear`` + ``RotationProto``
     - ✅
   * - STQ1_0 (Sherry)
     - 1.3125
     - ``StructuredBlock``
     - ✅
   * - TQ1_0
     - 1.6
     - ``Codebook``
     - ✅
   * - TQ2_0
     - 2.0
     - ``Codebook``
     - ✅
   * - BitNet b1.58
     - 1.58
     - ``Codebook``
     - ✅
   * - ParetoQ (SEQ ternary)
     - 1.58–2.0
     - ``Function`` or ``Codebook``
     - ✅
   * - Tequila (ICLR 2026)
     - 1.58
     - ``Codebook``
     - ✅

QuantizedTensorProto
++++++++++++++++++++

A quantized tensor cannot rely on ``shape × sizeof(data_type)`` to compute
its storage size (sub-byte packing, block metadata, sparse outliers, etc.).
It carries its own byte size explicitly.

.. tab-set::

   .. tab-item:: Specialized proto

      .. code-block:: text

         message QuantizedTensorProto {
             repeated int64 dims = 1;       // logical shape of the tensor
             bytes raw_data = 2;            // quantized payload
             int64 n_bytes = 3;            // byte size of raw_data
             int32 quantized_type = 4;      // index into ModelProto.quantizations
             optional QuantizationProto quantization = 5;  // inline if unique to this tensor
             string name = 6;
             string doc_string = 7;
         }

   .. tab-item:: Custom types

      .. code-block:: text

         StructProto {
             type: <quantization-profile type index>
             raw_data: ...
             name: ...
             doc_string: ...
         }

``QuantizedTensorProto`` becomes ``StructProto``. ``raw_data``,
``external_data``, ``name``, and ``doc_string`` are carried by that generic
value. Its exact model-level or inline ``StructTypeProto`` replaces
``quantized_type`` and ``quantization``. Counts such as a block or tile
number are concrete dimensions of that type. The size computed from the
physical type must equal ``raw_data.size()`` or external-data ``length``.
Logical dimensions and element type belong to the decoder output
``ValueInfoProto``.

In both forms, ``name`` identifies the concrete value and ``doc_string``
documents it.

TypeProto and containers
^^^^^^^^^^^^^^^^^^^^^^^^

The following specialized container additions are therefore not required by
the recommended implementation. They document what would be necessary only
if ``QuantizedTensorProto`` remained a distinct value category:

.. tab-set::

   .. tab-item:: Specialized proto

      .. code-block:: text

         message TypeProto {
             message QuantizedTensor {
                 repeated int32 allowed_quantized_type = 1;
                 optional int32 elem_type = 2;       // decoded logical element type
                 optional TensorShapeProto shape = 3;
             }

             oneof value {
                 ...
                 QuantizedTensor quantized_tensor_type = <N>;
             }
         }

         message SequenceProto {
             enum DataType {
                 ...
                 QUANTIZED_TENSOR = <N>;
             }
             repeated QuantizedTensorProto quantized_tensor_values = <N>;
             optional TypeProto value_type = <N+1>;
         }

         message OptionalProto {
             enum DataType {
                 ...
                 QUANTIZED_TENSOR = <N>;
             }
             optional QuantizedTensorProto quantized_tensor_value = <N>;
             optional TypeProto value_type = <N+1>;
         }

   .. tab-item:: Custom types

      .. code-block:: text

         message TypeProto {
             oneof value {
                 ...
                 StructTypeProto struct_type = <N>;
             }
         }

         message SequenceProto {
             enum DataType {
                 ...
                 STRUCT = <N>;
             }
             repeated StructProto struct_values = <N>;
             optional TypeProto value_type = <N+1>;
         }

         message OptionalProto {
             enum DataType {
                 ...
                 STRUCT = <N>;
             }
             optional StructProto struct_value = <N>;
             optional TypeProto value_type = <N+1>;
         }

``allowed_quantized_type`` is a set of indices into
``ModelProto.quantizations``. An empty set accepts any quantization declaration
whose decoded element type and logical shape satisfy the remaining
constraints. This allows a sequence or map to contain pages using different
quantization formats without becoming untyped.

The corresponding value containers require a new category.

These specialized branches map to the generic integrations defined by
:ref:`l-next-steps-custom-types`:

* ``TypeProto.QuantizedTensor`` becomes
  ``TypeProto.struct_type``;
* ``QUANTIZED_TENSOR`` becomes the ``STRUCT`` value category;
* ``quantized_tensor_values`` and ``quantized_tensor_value`` become
  ``struct_values`` and ``struct_value``;
* heterogeneous pages use the unconstrained static structured category,
  while each ``StructProto`` carries its exact physical type.

``MapProto`` inherits support because its values are represented by a
``SequenceProto``. Consequently, ``Sequence<QuantizedTensor>`` and
``Map<int64, QuantizedTensor>`` can represent a paged KV-cache with a different
quantization declaration for each page.

``value_type`` is required for these new categories and carries the complete
``TypeProto.QuantizedTensor`` constraint. It makes standalone sequence and map
values self-describing and must agree with any enclosing ``ValueInfoProto``.

The generalized design in :ref:`l-next-steps-custom-types` avoids duplicating
this machinery: a quantized tensor is an structured value with an explicit
physical type and a decoder with a typed output signature.

QuantizationProto
+++++++++++++++++

.. tab-set::

   .. tab-item:: Specialized proto

      .. code-block:: text

         // Describes how a tensor is quantized. Nine variants cover the
         // common families without separate messages for scalar/vector
         // codebooks, one-/multi-dimensional tiling, identity casts, or
         // packed linear layouts.
         // TilingQuantizationProto can nest QuantizationProto to express
         // multi-level hierarchies (e.g. K-Quants, MXFP).
         // Optional pre/post rotations support QuIP#, SmoothQuant, etc.
         message QuantizationProto {
             oneof kind {
                 LinearUniformProto linear = 1;
                 CodebookProto codebook = 2;
                 FloatingPointUniformProto floating_point = 4;
                 SparseQuantizationProto sparse = 5;
                 LogUniformProto log = 6;
                 FunctionUniformProto function = 7;
                 TilingQuantizationProto tiling = 13;
                 CastUniformProto cast = 16;
                 StructuredBlockUniformProto structured_block = 18;
             }
             int32 data_type = 15;              // dequantized element type (same enum as TensorProto.data_type)
             string doc_string = 9;             // human-readable description
             repeated StringStringEntryProto metadata_props = 10;  // arbitrary key-value metadata
             optional RotationProto pre_rotation = 11;   // rotation applied before quantization
             optional RotationProto post_rotation = 12;  // rotation applied after dequantization
         }

   .. tab-item:: Custom types

      .. code-block:: text

         ModelProto {
             struct_types: [
                 StructTypeProto {
                     // One concrete array or structure and its decoder.
                 },
                 ...
             ]
         }

There is no direct ``QuantizationProto`` message in the recommended schema.
Each entry becomes one concrete ``StructTypeProto`` in
``ModelProto.struct_types``:

* the selected profile determines its ``array`` or ``structure``;
* scalar parameters and lookup tables become fields with ``constant``;
* ``data_type`` and the logical shape are declared by the decoder output
  ``ValueInfoProto``;
* pre/post rotations become decoder or encoder nodes;
* ``doc_string`` and ``metadata_props`` remain type documentation and
  metadata.

The translation is lossless: every specialized field must map to serialized
structure, a named constant field, decoder/encoder structure, or metadata. A
decoder must not hide a profile parameter that was explicit in the
specialized declaration.

Code blocks containing names such as ``N``, ``K``, ``packed_bytes``, or
``tile_type`` are translation templates, not valid serialized declarations.
Every such name is replaced by a concrete value or model type index in an
``StructTypeProto``.

LinearUniformProto
^^^^^^^^^^^^^^^^^^

Classic affine/symmetric: ``value = (q - zero_point) * scale``.

.. tab-set::

   .. tab-item:: Specialized proto

      .. code-block:: text

         message LinearUniformProto {
             int32 storage_type = 1;       // quantized element type (same enum as TensorProto.data_type)
             int32 bits = 2;               // number of bits (e.g. 4, 8)
             bool symmetric = 3;           // true if zero_point is always 0
             oneof scale {
                 float scale_float = 4;    // scale as float
                 int32 scale_int = 5;      // scale as shared exponent (value = q * 2^scale_int)
             }
             int64 zero_point = 6;         // quantization zero point
             int32 axis = 7;               // axis for per-channel, -1 if per-tensor
         }

   .. tab-item:: Custom types

      .. code-block:: text

         StructTypeProto {
             name: "LINEAR"
             structure: Structure {
                 field: { name: "values",       type: array(INT4, dimension=N) }
                 field: { name: "storage_type", constant: tensor(INT32, [], INT4) }
                 field: { name: "bits",         constant: tensor(INT32, [], 4) }
                 field: { name: "symmetric",    constant: tensor(BOOL, [], symmetric) }
                 field: { name: "scale",        constant: tensor(FLOAT, [], s) }
                 field: { name: "zero_point",   constant: tensor(INT64, [], z) }
                 field: { name: "axis",         constant: tensor(INT32, [], axis) }
             }
             decoder: FunctionProto {
                 // Y = (Cast(values) - zero_point) * scale
             }
         }

The physical codes are an ``Array``. When ``bits`` equals the canonical width
of ``storage_type``, the element type is used directly. Otherwise the packed
region is an ``Array(UINT8)`` and the decoder extracts exactly ``bits`` per
logical code. Scale and zero point are constant fields when shared, or
serialized array fields when they vary by channel or block.

``N`` is always replaced by the concrete physical count.
The displayed ``scale`` field represents ``scale_float``. A declaration using
``scale_int`` gives that field type ``INT32`` instead, preserving which member
of the specialized ``oneof`` was selected.

.. code-block:: python

    # Dequantize
    values = unpack(data, q.bits)
    if q.scale_float:
        result = (values - q.zero_point) * q.scale_float
    else:
        result = (values - q.zero_point) * (2 ** q.scale_int)

    # Quantize
    if q.scale_float:
        values = round(tensor / q.scale_float) + q.zero_point
    else:
        values = round(tensor / (2 ** q.scale_int)) + q.zero_point
    values = clip(values, 0, (1 << q.bits) - 1)
    raw_data = pack(values, q.bits)

CodebookProto
^^^^^^^^^^^^^

Scalar or additive vector lookup-table quantization. Scalar codebooks
use ``num_codebooks = vector_size = 1``. For additive vector
quantization, each vector is reconstructed as the sum of one entry from
each codebook.

.. tab-set::

   .. tab-item:: Specialized proto

      .. code-block:: text

         message CodebookProto {
             oneof scale {
                 float scale_float = 1;    // scale as float
                 int32 scale_int = 2;      // scale as shared exponent
             }
             repeated float codebook_data = 3;  // concatenated codebooks
             int32 packed_count = 4;       // number of values packed
             int32 packed_bytes = 5;       // into this many bytes
             int32 num_codebooks = 6;      // additive codebooks; 0 means 1
             int32 codebook_size = 7;      // entries per codebook; 0 means infer
             int32 vector_size = 8;        // values per entry; 0 means 1
             int32 index_bits = 9;         // bits per index; 0 means base-N packing
         }

   .. tab-item:: Custom types

      .. code-block:: text

         StructTypeProto {
             name: "CODEBOOK"
             structure: Structure {
                 field: { name: "indices",  type: array(UINT8, dimension=packed_bytes) }
                 field: {
                     name: "codebook"
                     constant: tensor(
                         FLOAT,
                         [codebook_value_count],
                         codebook_data
                     )
                 }
                 field: { name: "scale", constant: tensor(FLOAT, [], scale) }
                 field: { name: "packed_count", constant: tensor(INT32, [], packed_count) }
                 field: { name: "packed_bytes", constant: tensor(INT32, [], packed_bytes) }
                 field: { name: "num_codebooks", constant: tensor(INT32, [], num_codebooks) }
                 field: { name: "codebook_size", constant: tensor(INT32, [], codebook_size) }
                 field: { name: "vector_size", constant: tensor(INT32, [], vector_size) }
                 field: { name: "index_bits", constant: tensor(INT32, [], index_bits) }
             }
             decoder: FunctionProto {
                 // Y = scale * additive_gather(codebook, unpack(indices))
             }
         }

Packed indices are serialized as an ``Array(UINT8)``. The codebooks and scale
are constant fields, so they are stored once in the type and consume no
payload bytes. The decoder unpacks each index, gathers the selected scalar or
vector entry, sums additive codebooks, and applies the scale.

As for linear quantization, the ``scale`` field is ``FLOAT`` for
``scale_float`` and ``INT32`` for ``scale_int``.

.. code-block:: python

    # Dequantize
    num_codebooks = q.num_codebooks or 1
    vector_size = q.vector_size or 1
    codebook_size = q.codebook_size or (
        len(q.codebook_data) // (num_codebooks * vector_size))
    indices = unpack_indices(data, q.index_bits, codebook_size,
                             q.packed_count, q.packed_bytes)
    result = zeros(n_vectors * vector_size)
    for k in range(num_codebooks):
        codebook_k = q.codebook_data[
            k * codebook_size * vector_size:
            (k + 1) * codebook_size * vector_size]
        for i, idx in enumerate(indices[k]):
            begin = idx * vector_size
            result[i*vector_size:(i+1)*vector_size] += (
                codebook_k[begin:begin + vector_size])
    result *= q.scale

    # Quantize
    indices = nearest_additive_codewords(tensor / q.scale, q)
    raw_data = pack_indices(indices, q.index_bits, codebook_size,
                            q.packed_count, q.packed_bytes)

FloatingPointUniformProto
^^^^^^^^^^^^^^^^^^^^^^^^^

For finite normal values, micro-float quantization uses
``value = (-1)^sign * 2^(exp - bias) * (1 + mantissa)``.
Covers FP6, FP4, MXFP and similar reduced-precision floating-point formats.

.. tab-set::

   .. tab-item:: Specialized proto

      .. code-block:: text

         message FloatingPointUniformProto {
             int32 sign_bits = 1;          // sign bits (usually 1)
             int32 exponent_bits = 2;      // exponent bits (e.g. 3 for E3M2, 2 for E2M1)
             int32 mantissa_bits = 3;      // mantissa bits (e.g. 2 for E3M2, 1 for E2M1)
             int32 exponent_bias = 4;      // exponent bias (e.g. 3 for E3M2)
             bool has_inf = 5;             // true if format supports infinity
             bool has_nan = 6;             // true if format supports NaN
             bool split_storage = 7;       // true if sign+exp and mantissa stored separately (TC-FPn)
             int32 packed_count = 8;       // number of values packed
             int32 packed_bytes = 9;       // into this many bytes
         }

   .. tab-item:: Custom types

      .. code-block:: text

         StructTypeProto {  // translation template; not serialized as-is
             structure: Structure {
                 // Serialized payload.
                 field: {
                     name: "packed"
                     type: bit_packing(
                         dimension=packed_count,
                         components=[
                             sign:sign_bits,
                             exponent:exponent_bits,
                             mantissa:mantissa_bits
                         ]
                     )
                 }

                 // Type-level parameters: these consume no payload bytes.
                 field: {
                     name: "exponent_bias"
                     constant: tensor(INT32, [], exponent_bias)
                 }
                 field: {
                     name: "has_inf"
                     constant: tensor(BOOL, [], has_inf)
                 }
                 field: {
                     name: "has_nan"
                     constant: tensor(BOOL, [], has_nan)
                 }
             }
             decoder: FunctionProto {
                 // unpack packed_count values from packed_bytes bytes
                 // interpret sign/exponent/mantissa using the declared constants
                 // apply exceptional-value policy and reconstruct Y
             }
         }

         StructTypeProto {
             name: "FLOAT6_E3M2"
             structure: Structure {
                 field: {
                     name: "packed"
                     type: bit_packing(
                         dimension=4,
                         components=[sign:1, exponent:3, mantissa:2]
                     )
                 }
                 field: { name: "exponent_bias", constant: tensor(INT32, [], 3) }
                 field: { name: "has_inf", constant: tensor(BOOL, [], true) }
                 field: { name: "has_nan", constant: tensor(BOOL, [], true) }
             }
             decoder: FunctionProto {
                 output: "Y"
                 value_info: {
                     name: "Y"
                     type: TypeProto {
                         tensor_type: Tensor {
                             elem_type: FLOAT
                             shape: TensorShapeProto {
                                 dim: { dim_value: 4 }
                             }
                         }
                     }
                 }
                 // unpack four 6-bit values and decode E3M2 with exponent bias 3
             }
         }

A standard ONNX low-precision type such as ``FLOAT4E2M1`` or a float8 type is
an ordinary physical ``Array``. A non-standard width uses ``BitPacking`` so
the component widths and packed element count remain explicit. Split storage
is a ``Structure`` with one field per bit plane or component. Parameters not
already expressed by the physical structure remain named constant fields.
Thus ``sign_bits``, ``exponent_bits``, ``mantissa_bits``, ``packed_count``,
``packed_bytes``, and ``split_storage`` are checked from the physical layout
rather than duplicated as constants.

This is a generic profile template, not a serializable element type:
``sign_bits``, ``exponent_bits``, and the other symbolic values are replaced
when the concrete physical type is declared. A generic ``IEEE_FLOAT`` cannot
be a ``TensorProto.data_type`` or ``TypeProto.Tensor.elem_type`` because it
does not determine the width or the decoder's standard ONNX output type.

The following concrete ``FLOAT6_E3M2`` declaration packs four 6-bit values
into three bytes. It uses an IEEE-like exceptional-value policy, fixes every
profile parameter with ``Field.constant``, and decodes to a standard ONNX
``FLOAT`` tensor:

When ``split_storage`` is true, the serialized ``packed`` field is replaced
by explicitly named component arrays such as ``sign_exponent`` and
``mantissa``. The constant remains useful to profiles and validation, while
the physical structure itself is sufficient to locate the components.

.. code-block:: python

    # Dequantize
    bits_per_elem = q.sign_bits + q.exponent_bits + q.mantissa_bits
    raw_values = unpack(data, bits_per_elem, q.split_storage)
    result = [fp_decode(v, q.sign_bits, q.exponent_bits,
                        q.mantissa_bits, q.exponent_bias)
              for v in raw_values]

SparseQuantizationProto
^^^^^^^^^^^^^^^^^^^^^^^

Sparse + dense decomposition (SpQR, SqueezeLLM). Outlier values above
a threshold are stored separately in higher precision; the rest uses
a base quantization scheme.

.. tab-set::

   .. tab-item:: Specialized proto

      .. code-block:: text

         message SparseQuantizationProto {
             QuantizationProto base_quant = 1;  // quantization for non-outlier values
             int32 outlier_data_type = 2;       // data type for outlier values (e.g. FLOAT16)
             float outlier_threshold = 3;       // absolute value threshold for outlier detection
             float outlier_ratio = 4;           // fraction of values stored as outliers (e.g. 0.01)
         }

   .. tab-item:: Custom types

      .. code-block:: text

         StructTypeProto {
             name: "SPARSE_QUANTIZED"
             structure: Structure {
                 field: { name: "base",    type: struct(type_index=base_type) }
                 field: { name: "indices", type: array(INT64, dimension=K) }
                 field: { name: "values",  type: array(FLOAT16, dimension=K) }
                 field: {
                     name: "outlier_threshold"
                     constant: tensor(FLOAT, [], outlier_threshold)
                 }
                 field: {
                     name: "outlier_ratio"
                     constant: tensor(FLOAT, [], outlier_ratio)
                 }
             }
             decoder: FunctionProto {
                 // Y = scatter(decode(base), indices, values)
             }
         }

The dense/base region is a referenced structured physical type. Outlier
indices and values are concrete arrays in the same root ``Structure``.
The concrete element type of ``values`` preserves ``outlier_data_type``.
Threshold and ratio guide the encoder but are not required to decode an
already serialized value; they remain explicit constants. The root decoder
incorporates the base decoder nodes explicitly; nested decoders are not
invoked implicitly.

``K`` is a concrete outlier count in each physical declaration.

.. code-block:: python

    # Dequantize
    base_values = dequantize(base_data, q.base_quant)
    outlier_indices, outlier_values = read_sparse(outlier_data, q.outlier_data_type)
    result = base_values
    result[outlier_indices] = outlier_values

    # Quantize
    mask = abs(tensor) > q.outlier_threshold
    outlier_indices, outlier_values = where(mask), tensor[mask]
    base_tensor = tensor.copy(); base_tensor[mask] = 0
    base_data = quantize(base_tensor, q.base_quant)
    outlier_data = write_sparse(outlier_indices, outlier_values, q.outlier_data_type)

LogUniformProto
^^^^^^^^^^^^^^^

Logarithmic quantization: ``value = sign * base^(q + offset)``.

.. tab-set::

   .. tab-item:: Specialized proto

      .. code-block:: text

         message LogUniformProto {
             int32 bits = 1;               // number of bits (e.g. 4, 8)
             float base = 2;              // logarithm base (e.g. 2.0)
             float offset = 3;            // exponent offset
             bool has_sign = 4;           // true if sign bit is stored separately
         }

   .. tab-item:: Custom types

      .. code-block:: text

         StructTypeProto {
             name: "LOG_UNIFORM"
             structure: Structure {
                 field: { name: "codes",  type: array(UINT4, dimension=N) }
                 field: { name: "signs",  type: array(UINT8, dimension=sign_bytes) }
                 field: { name: "bits", constant: tensor(INT32, [], 4) }
                 field: { name: "base", constant: tensor(FLOAT, [], base) }
                 field: { name: "offset", constant: tensor(FLOAT, [], offset) }
                 field: { name: "has_sign", constant: tensor(BOOL, [], has_sign) }
             }
             decoder: FunctionProto {
                 // Y = sign * Pow(base, Cast(codes) + offset)
             }
         }

Codes use a native fixed-width ``Array`` when possible and packed
``Array(UINT8)`` otherwise. A separate sign plane is another array field.
``base`` and ``offset`` are constants consumed by the decoder.

The ``signs`` physical field is omitted when ``has_sign`` is false.

.. code-block:: python

    # Dequantize
    values = unpack(data, q.bits)
    if q.has_sign:
        signs = extract_sign_bits(values)
        exponents = extract_magnitude(values)
        result = signs * (q.base ** (exponents + q.offset))
    else:
        result = q.base ** (values + q.offset)

    # Quantize
    if q.has_sign:
        signs = sign(tensor)
        exponents = round(log(abs(tensor)) / log(q.base) - q.offset)
        values = pack_sign_magnitude(signs, exponents)
    else:
        values = round(log(tensor) / log(q.base) - q.offset)
    values = clip(values, 0, (1 << q.bits) - 1)
    raw_data = pack(values, q.bits)

FunctionUniformProto
^^^^^^^^^^^^^^^^^^^^

Custom quantization/dequantization defined by op names.
``storage_type`` and ``bits`` describe the storage format of the quantized
data so the runtime can read the raw bytes correctly. The custom ops
referenced by ``quantize_op`` and ``dequantize_op`` handle the actual
conversion logic. The dequantize op takes a tensor of ``storage_type``
and returns the type specified by ``QuantizationProto.data_type``;
the quantize op does the reverse.

.. tab-set::

   .. tab-item:: Specialized proto

      .. code-block:: text

         message FunctionUniformProto {
             int32 storage_type = 1;       // storage element type (how raw bytes are interpreted)
             int32 bits = 2;               // bits per element (for sub-byte packing)
             string quantize_op = 3;       // op: data_type -> storage_type (e.g. "custom::Quantize")
             string dequantize_op = 4;     // op: storage_type -> data_type (e.g. "custom::Dequantize")
             optional BlockLayoutProto block_layout = 5;  // physical block structure (for introspection)
         }

   .. tab-item:: Custom types

      .. code-block:: text

         StructTypeProto {
             name: "FUNCTION_QUANTIZED"
             structure: Structure {
                 field: {
                     name: "payload"
                     type: array(UINT8, dimension=packed_bytes)
                 }
                 field: {
                     name: "storage_type"
                     constant: tensor(INT32, [], storage_type)
                 }
                 field: { name: "bits", constant: tensor(INT32, [], bits) }
             }
             decoder: FunctionProto { ... }  // standard-domain form, when available
             encoder: FunctionProto { ... }
             metadata_props: { key: "quantize_op", value: quantize_op }
             metadata_props: { key: "dequantize_op", value: dequantize_op }
         }

``storage_type``, ``bits``, and ``block_layout`` become the concrete
``Array``/``Structure`` schema. ``dequantize_op`` becomes
``StructTypeProto.decoder`` and ``quantize_op`` becomes ``encoder`` when
both can be expressed as deterministic standard-domain ONNX functions.

If either operation requires a custom-domain operator, the physical type
remains valid but omits that portable function. The consuming custom operator
is then responsible for interpretation. The original operator identifiers are
preserved as ``quantize_op`` and ``dequantize_op`` metadata so the translation
remains lossless.

When ``block_layout`` is present, its concrete fields replace the raw
``payload`` array and determine the same exact byte size.

.. code-block:: python

    # Dequantize
    if q.block_layout:
        blocks = split_blocks(data, q.block_layout.bytes_per_block)
        result = concat([call_op(q.dequantize_op, block) for block in blocks])
    else:
        raw = interpret(data, q.storage_type, q.bits)
        result = call_op(q.dequantize_op, raw)

    # Quantize
    if q.block_layout:
        blocks = tile(tensor, q.block_layout.block_size)
        raw_data = concat([call_op(q.quantize_op, block) for block in blocks])
    else:
        raw_data = call_op(q.quantize_op, tensor)

TilingQuantizationProto
^^^^^^^^^^^^^^^^^^^^^^^

Recursive one- or multi-dimensional block quantization. Weights are
partitioned into tiles along ``axes``. An empty ``axes`` denotes the
flattened tensor, making classic one-dimensional block quantization a
special case. If there are more tiles than entries in ``elem_quant``,
the list is cycled from the beginning
(``elem_quant[i % len(elem_quant)]``). Nesting supports multi-level
hierarchies such as K-Quants and MXFP.

.. tab-set::

   .. tab-item:: Specialized proto

      .. code-block:: text

         message TilingQuantizationProto {
             repeated int64 tile_shape = 1;                 // tile size; one value if axes is empty
             repeated int32 axes = 2;                       // empty = flattened tensor
             repeated QuantizationProto elem_quant = 3;     // cycled over tiles
             repeated int32 perm = 4;                       // permutation of axes in memory layout
         }

   .. tab-item:: Custom types

      .. code-block:: text

         StructTypeProto {
             name: "TILED"
             structure: Structure {
                 field: {
                     name: "tiles"
                     type: array(
                         struct(type_index=tile_type),
                         dimension=concrete_tile_count
                     )
                 }
                 field: {
                     name: "tile_shape"
                     constant: tensor(INT64, [tile_rank], concrete_tile_shape)
                 }
                 field: {
                     name: "axes"
                     constant: tensor(INT32, [axis_count], concrete_axes)
                 }
                 field: {
                     name: "perm"
                     constant: tensor(INT32, [perm_rank], concrete_perm)
                 }
             }
             decoder: FunctionProto {
                 // decode each tile, place it at concrete coordinates, inverse perm
             }
         }

A homogeneous tiling stores a nested ``Array`` whose element references the
concrete tile type. The root is a ``Structure`` so concrete ``tile_shape``,
``axes``, and ``perm`` values remain explicit constant fields. If tiles cycle
through different quantizations, the root has one physical field per concrete
tile or homogeneous tile region. The decoder concatenates or scatters tile
outputs and incorporates the tile decoder nodes explicitly. No symbolic tile
count, modulo rule, or implicit nested-decoder invocation remains in the
physical type.

.. code-block:: python

    # Dequantize
    shape = resolve_tile_shape(q.tile_shape, q.axes, tensor_shape)
    tiles = split_tiles(data, shape, q.axes, q.perm)
    result = empty(tensor_shape)
    for i, (coords, tile_data) in enumerate(tiles):
        eq = q.elem_quant[i % len(q.elem_quant)]
        result[coords] = dequantize(tile_data, eq)
    if q.perm:
        result = inverse_permute(result, q.perm)

CastUniformProto
^^^^^^^^^^^^^^^^

Type conversion without quantization. Stores values cast from the
original type to a different type (e.g. float32 → bfloat16).
The source type is ``QuantizationProto.data_type``, the target type
is ``storage_type``. Setting ``storage_type`` equal to ``data_type`` is
the identity operation, useful for pure retiling.

.. tab-set::

   .. tab-item:: Specialized proto

      .. code-block:: text

         message CastUniformProto {
             int32 storage_type = 1;        // target element type (same enum as TensorProto.data_type)
         }

   .. tab-item:: Custom types

      .. code-block:: text

         StructTypeProto {
             name: "CAST"
             array: Array {
                 element_type: BFLOAT16
                 dimension: N
             }
             decoder: FunctionProto {
                 // Y = Cast(values, original_type)
             }
         }

The payload is a concrete ``Array`` of ``storage_type``. The decoder output
``ValueInfoProto`` declares the original type and shape; the decoder contains
the standard ONNX ``Cast`` and any required reshape.

.. code-block:: python

    # Dequantize
    result = cast(data, from_type=q.storage_type, to_type=quant_proto.data_type)

    # Quantize
    raw_data = cast(tensor, from_type=quant_proto.data_type, to_type=q.storage_type)

StructuredBlockUniformProto
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Generic block quantization with explicit physical layout. Each block
contains named fields at known bit offsets. ``VALUES`` can be decoded
directly for packed linear formats, or an index formula can combine
fields into codebook indices. Optional scale, zero point, bias, and
scatter fields cover prepacked linear formats (CompInt8, QGEMM), exotic
formats (STQ1_0), and future layouts while keeping dequantization fully
deducible from the proto structure.

.. tab-set::

   .. tab-item:: Specialized proto

      .. code-block:: text

         enum BlockFieldRole {
             VALUES = 0;           // quantized values or codebook indices
             SIGN = 1;             // sign bits
             SCALE = 2;            // per-block scale
             ZERO_POINT = 3;       // per-block zero point
             BIAS = 4;             // per-block bias correction
             EXPONENT = 5;         // shared exponent
             CODE = 6;             // codebook index (partial)
             MASK = 7;             // sparsity mask
         }

         message BlockFieldProto {
             BlockFieldRole role = 1;       // semantic role of this field
             int32 bit_offset = 2;          // offset in block or field record
             int32 bit_width = 3;           // bits per element
             int32 count = 4;               // number of elements
             int32 data_type = 5;           // element type (0 = raw unsigned bits)
         }

         message BlockLayoutProto {
             int32 block_size = 1;                  // logical values per block
             int32 bytes_per_block = 2;             // physical bytes per block
             repeated BlockFieldProto fields = 3;   // physical fields in the block
         }

         enum BlockStorageOrder {
             INTERLEAVED = 0;       // all fields for block 0, then all fields for block 1
             SEQUENTIAL = 1;        // one region per field, each containing all blocks
         }

         message FieldWeightProto {
             BlockFieldRole field = 1;      // which field to use
             int32 multiplier = 2;          // coefficient in index formula
         }

         message ScatterProto {
             int32 group_size = 1;          // logical group size (e.g. 64)
             int32 vector_size = 2;         // lanes per codebook entry (e.g. 4)
             int32 stride = 3;             // stride within group (e.g. 16)
         }

         message StructuredBlockUniformProto {
             BlockLayoutProto block_layout = 1;             // physical block structure
             repeated float codebook_data = 2;              // empty = use VALUES directly
             int32 codebook_vector_size = 3;                // values per codebook entry
             repeated FieldWeightProto index_formula = 4;   // index = sum(field * multiplier)
             optional ScatterProto scatter = 5;             // output element placement (absent = contiguous)
             BlockStorageOrder storage_order = 6;           // physical organization of block fields
             optional int32 axis = 7;                       // absent = flattened tensor
         }

   .. tab-item:: Custom types

      .. code-block:: text

         StructTypeProto {
             name: "STRUCTURED_BLOCK"
             structure: Structure {
                 field: {
                     name: "blocks"
                     type: array(
                         struct(type_index=block_type),
                         dimension=concrete_block_count
                     )
                 }
                 field: {
                     name: "codebook"
                     constant: tensor(
                         FLOAT,
                         [codebook_value_count],
                         codebook_data
                     )
                 }
                 field: {
                     name: "codebook_vector_size"
                     constant: tensor(INT32, [], codebook_vector_size)
                 }
                 field: {
                     name: "index_fields"
                     constant: tensor(
                         INT32,
                         [index_term_count],
                         concrete_index_fields
                     )
                 }
                 field: {
                     name: "index_multipliers"
                     constant: tensor(
                         INT32,
                         [index_term_count],
                         concrete_index_multipliers
                     )
                 }
                 field: {
                     name: "storage_order"
                     constant: tensor(INT32, [], concrete_storage_order)
                 }
                 field: {
                     name: "block_size"
                     constant: tensor(INT32, [], concrete_block_size)
                 }
                 field: {
                     name: "axis"
                     constant: tensor(INT32, [], concrete_axis)
                 }
             }
             decoder: FunctionProto {
                 // index = sum(named_field * multiplier)
                 // Y = scatter(codebook[index] * scale + bias)
             }
         }

This profile maps directly to the generic mechanism:

* ``BlockFieldProto`` becomes a named ``Structure.Field``;
* ``count`` becomes ``Array.dimension``;
* ``data_type`` becomes the scalar array element type;
* field roles become ordinary field names used by the decoder;
* explicit padding replaces ``bit_offset`` and ``bytes_per_block``;
* ``codebook_data`` becomes a constant field;
* ``index_formula`` and ``ScatterProto`` become constant parameters consumed
  by decoder nodes;
* ``INTERLEAVED`` is an array of block structures;
* ``SEQUENTIAL`` is a structure containing one array per field.

The ``codebook``, index-formula, and ``axis`` fields are omitted when their
specialized counterparts are absent. When ``scatter`` is present, its three
integer members are additional constant fields named ``scatter_group_size``,
``scatter_vector_size``, and ``scatter_stride``. The concrete block type
itself preserves ``block_size``, field widths and counts, explicit padding,
and therefore ``bytes_per_block``.

No separate ``BlockFieldProto``, ``BlockLayoutProto``, ``FieldWeightProto``,
or ``ScatterProto`` is implemented.

.. code-block:: python

    # Dequantize
    blocks = parse_structured_blocks(
        data, q.block_layout, q.storage_order, q.axis, tensor_shape)
    result = []
    for block in blocks:
        fields = parse_fields(block, q.block_layout.fields)
        scale = fields[SCALE][0] if SCALE in fields else 1.0
        zp = fields[ZERO_POINT][0] if ZERO_POINT in fields else 0.0
        if q.codebook_data:
            assert q.index_formula, "index_formula is required when codebook_data is set"
            assert q.codebook_vector_size > 0, "codebook_vector_size must be > 0"
            n_entries = len(q.codebook_data) // q.codebook_vector_size
            values = []
            n_codes = None
            for fw in q.index_formula:
                assert fw.field in fields, f"missing field {fw.field} for index_formula"
                if n_codes is None:
                    n_codes = len(fields[fw.field])
                else:
                    assert len(fields[fw.field]) == n_codes, f"field {fw.field} has inconsistent length (expected {n_codes})"
            assert n_codes is not None, "index_formula must not be empty"
            for i in range(n_codes):
                raw_idx = sum(fields[fw.field][i] * fw.multiplier for fw in q.index_formula)
                idx = int(raw_idx)
                assert raw_idx == idx, f"codebook index {raw_idx} is not an integer"
                assert 0 <= idx < n_entries, f"codebook index {idx} out of range"
                begin = idx * q.codebook_vector_size
                values.extend(q.codebook_data[begin:begin + q.codebook_vector_size])
        else:
            values = fields[VALUES]
        if q.scatter:
            values = inverse_scatter(values, q.scatter)
        bias = fields[BIAS][0] if BIAS in fields else 0.0
        result.append((values - zp) * scale + bias)
    result = concat(result)

RotationProto
^^^^^^^^^^^^^

Pre/post rotation applied to the tensor (QuIP#, SmoothQuant).
Dequantization with rotation: ``values = post_rotation @ dequant(data) @ pre_rotation``.

.. tab-set::

   .. tab-item:: Specialized proto

      .. code-block:: text

         enum RotationType {
             HADAMARD = 0;
             PLAIN = 1;
         }

         message RotationProto {
             RotationType matrix_type = 1;         // type of rotation
             repeated int32 dims = 2;              // shape of the rotation matrix
             optional int32 matrix_index = 3;      // index into ModelProto.rotation_matrices (for PLAIN)
         }

   .. tab-item:: Custom types

      .. code-block:: text

         field: {
             name: "rotation"
             constant: tensor(FLOAT, concrete_rotation_dims, concrete_matrix)
         }
         field: {
             name: "rotation_type"
             constant: tensor(INT32, [], PLAIN)
         }

         decoder: FunctionProto {
             // Y = MatMul(rotation, decode_physical_fields(...))
         }

Rotation is not a physical root type. A plain matrix is a field with
``constant`` in the quantized type and matrix multiplication is part of the
decoder or encoder. A Hadamard rotation needs no stored matrix and is expressed
directly by standard ONNX nodes. ``ModelProto.rotation_matrices`` and
``matrix_index`` are therefore resolved rather than copied into the generic
type.

Known quantization schemes
+++++++++++++++++++++++++++

The examples below retain the specialized profile notation because it is
compact and makes comparison with existing formats easy. They are not
additional wire messages. Each example is implemented by applying the
corresponding custom-type translation above, with all dimensions and payload
sizes made concrete.

INT8 symmetric (per-tensor)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

    QuantizationProto {
        linear: LinearUniformProto {
            storage_type: INT8, bits: 8, symmetric: true,
            scale_float: 0.02, zero_point: 0, axis: -1
        }
    }

INT4 symmetric (per-tensor)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a tensor of 1024 elements:

.. tab-set::

   .. tab-item:: Specialized proto

      .. code-block:: text

         QuantizationProto {
             linear: LinearUniformProto {
                 storage_type: INT4
                 bits: 4
                 symmetric: true
                 scale_float: 0.02
                 zero_point: 0
                 axis: -1
             }
         }

   .. tab-item:: Custom types

      .. code-block:: text

         StructTypeProto {
             name: "INT4_SYMMETRIC_1024"
             structure: Structure {
                 field: {
                     name: "values"
                     type: array(INT4, dimension=1024)
                 }
                 field: {
                     name: "storage_type"
                     constant: tensor(INT32, [], INT4)
                 }
                 field: { name: "bits", constant: tensor(INT32, [], 4) }
                 field: { name: "symmetric", constant: tensor(BOOL, [], true) }
                 field: { name: "scale", constant: tensor(FLOAT, [], 0.02) }
                 field: { name: "zero_point", constant: tensor(INT64, [], 0) }
                 field: { name: "axis", constant: tensor(INT32, [], -1) }
             }
         }

INT4 asymmetric (GPTQ, per-group of 128)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a tensor of 1024 elements (8 groups):

.. tab-set::

   .. tab-item:: Specialized proto

      .. code-block:: text

         QuantizationProto {
             tiling: TilingQuantizationProto {
                 tile_shape: [128]
                 elem_quant: [
                     QuantizationProto { linear: LinearUniformProto {
                         storage_type: INT4, bits: 4, symmetric: false,
                         scale_float: 0.015, zero_point: 8, axis: -1
                     }},
                     QuantizationProto { linear: LinearUniformProto {
                         storage_type: INT4, bits: 4, symmetric: false,
                         scale_float: 0.012, zero_point: 7, axis: -1
                     }},
                     QuantizationProto { linear: LinearUniformProto {
                         storage_type: INT4, bits: 4, symmetric: false,
                         scale_float: 0.014, zero_point: 8, axis: -1
                     }},
                     QuantizationProto { linear: LinearUniformProto {
                         storage_type: INT4, bits: 4, symmetric: false,
                         scale_float: 0.013, zero_point: 8, axis: -1
                     }},
                     QuantizationProto { linear: LinearUniformProto {
                         storage_type: INT4, bits: 4, symmetric: false,
                         scale_float: 0.016, zero_point: 7, axis: -1
                     }},
                     QuantizationProto { linear: LinearUniformProto {
                         storage_type: INT4, bits: 4, symmetric: false,
                         scale_float: 0.011, zero_point: 8, axis: -1
                     }},
                     QuantizationProto { linear: LinearUniformProto {
                         storage_type: INT4, bits: 4, symmetric: false,
                         scale_float: 0.015, zero_point: 9, axis: -1
                     }},
                     QuantizationProto { linear: LinearUniformProto {
                         storage_type: INT4, bits: 4, symmetric: false,
                         scale_float: 0.012, zero_point: 8, axis: -1
                     }}
                 ]
             }
         }

   .. tab-item:: Custom types

      .. code-block:: text

         StructTypeProto {
             name: "GPTQ_INT4_1024_GROUP128"
             structure: Structure {
                 field: {
                     name: "values"
                     type: array(INT4, dimensions=[8, 128])
                 }
                 field: {
                     name: "scales"
                     constant: tensor(
                         FLOAT,
                         [8],
                         [0.015, 0.012, 0.014, 0.013,
                          0.016, 0.011, 0.015, 0.012]
                     )
                 }
                 field: {
                     name: "zero_points"
                     constant: tensor(INT64, [8], [8, 7, 8, 8, 7, 8, 9, 8])
                 }
                 field: {
                     name: "tile_shape"
                     constant: tensor(INT64, [1], [128])
                 }
                 field: {
                     name: "storage_type"
                     constant: tensor(INT32, [], INT4)
                 }
                 field: { name: "bits", constant: tensor(INT32, [], 4) }
                 field: { name: "symmetric", constant: tensor(BOOL, [], false) }
                 field: { name: "axis", constant: tensor(INT32, [], -1) }
             }
         }

NF4 (QLoRA / bitsandbytes)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. tab-set::

   .. tab-item:: Specialized proto

      .. code-block:: text

         QuantizationProto {
             codebook: CodebookProto {
                 scale_float: 1.0,
                 codebook_data: [-1.0, -0.6962, -0.5251, -0.3949, -0.2844,
                            -0.1848, -0.0911, 0.0, 0.0796, 0.1609,
                            0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0],
                 packed_count: 2, packed_bytes: 1
             }
         }

   .. tab-item:: Custom types

      .. code-block:: text

         StructTypeProto {
             name: "NF4"
             structure: Structure {
                 field: {
                     name: "indices"
                     type: array(UINT8, dimension=packed_byte_count)
                 }
                 field: {
                     name: "codebook"
                     constant: tensor(
                         FLOAT,
                         [16],
                         [-1.0, -0.6962, -0.5251, -0.3949,
                          -0.2844, -0.1848, -0.0911, 0.0,
                          0.0796, 0.1609, 0.2461, 0.3379,
                          0.4407, 0.5626, 0.7230, 1.0]
                     )
                 }
                 field: { name: "scale", constant: tensor(FLOAT, [], 1.0) }
                 field: { name: "packed_count", constant: tensor(INT32, [], 2) }
                 field: { name: "packed_bytes", constant: tensor(INT32, [], 1) }
                 field: { name: "num_codebooks", constant: tensor(INT32, [], 1) }
                 field: { name: "codebook_size", constant: tensor(INT32, [], 16) }
                 field: { name: "vector_size", constant: tensor(INT32, [], 1) }
                 field: { name: "index_bits", constant: tensor(INT32, [], 0) }
             }
         }

AWQ INT4 (per-group of 128)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. tab-set::

   .. tab-item:: Specialized proto

      .. code-block:: text

         QuantizationProto {
             tiling: TilingQuantizationProto {
                 tile_shape: [128]
                 elem_quant: [
                     QuantizationProto { linear: LinearUniformProto {
                         storage_type: INT4, bits: 4,
                         symmetric: false, scale_float: 0.01, zero_point: 8, axis: -1
                     }},
                     // ... one per block
                 ]
             }
         }

   .. tab-item:: Custom types

      .. code-block:: text

         ModelProto {
             struct_types: [
                 StructTypeProto {
                     name: "AWQ_INT4_BLOCK_128"
                     structure: Structure {
                         field: { name: "values", type: array(INT4, dimension=128) }
                         field: {
                             name: "storage_type"
                             constant: tensor(INT32, [], INT4)
                         }
                         field: { name: "bits", constant: tensor(INT32, [], 4) }
                         field: { name: "symmetric", constant: tensor(BOOL, [], false) }
                         field: { name: "scale", constant: tensor(FLOAT, [], 0.01) }
                         field: { name: "zero_point", constant: tensor(INT64, [], 8) }
                         field: { name: "axis", constant: tensor(INT32, [], -1) }
                     }
                 },
                 StructTypeProto {
                     name: "AWQ_INT4_TILES"
                     structure: Structure {
                         field: {
                             name: "blocks"
                             type: array(
                                 struct(type_index=0),
                                 dimension=block_count
                             )
                         }
                         field: {
                             name: "tile_shape"
                             constant: tensor(INT64, [1], [128])
                         }
                     }
                 }
             ]
         }

Q2_K (llama.cpp, nested 256/16)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. tab-set::

   .. tab-item:: Specialized proto

      .. code-block:: text

         QuantizationProto {
             tiling: TilingQuantizationProto {
                 tile_shape: [256]
                 elem_quant: [
                     QuantizationProto {
                         tiling: TilingQuantizationProto {
                             tile_shape: [16]
                             elem_quant: [
                                 QuantizationProto { linear: LinearUniformProto {
                                     storage_type: UINT2, bits: 2,
                                     symmetric: false, scale_float: 0.005, zero_point: 1, axis: -1
                                 }},
                                 // ... 16 sub-blocks per super-block
                             ]
                         }
                     },
                     // ... one per super-block
                 ]
             }
         }

   .. tab-item:: Custom types

      .. code-block:: text

         ModelProto {
             struct_types: [
                 StructTypeProto {
                     name: "Q2_K_SUBBLOCK_16"
                     structure: Structure {
                         field: { name: "values", type: array(UINT8, dimension=4) }
                         field: {
                             name: "storage_type"
                             constant: tensor(INT32, [], UINT2)
                         }
                         field: { name: "bits", constant: tensor(INT32, [], 2) }
                         field: { name: "symmetric", constant: tensor(BOOL, [], false) }
                         field: { name: "scale", constant: tensor(FLOAT, [], 0.005) }
                         field: { name: "zero_point", constant: tensor(INT64, [], 1) }
                         field: { name: "axis", constant: tensor(INT32, [], -1) }
                     }
                 },
                 StructTypeProto {
                     name: "Q2_K_SUPERBLOCK_256"
                     structure: Structure {
                         field: {
                             name: "subblocks"
                             type: array(struct(type_index=0), dimension=16)
                         }
                         field: {
                             name: "tile_shape"
                             constant: tensor(INT64, [1], [16])
                         }
                     }
                 },
                 StructTypeProto {
                     name: "Q2_K_TENSOR"
                     structure: Structure {
                         field: {
                             name: "superblocks"
                             type: array(
                                 struct(type_index=1),
                                 dimension=superblock_count
                             )
                         }
                         field: {
                             name: "tile_shape"
                             constant: tensor(INT64, [1], [256])
                         }
                     }
                 }
             ]
         }

Q3_K (llama.cpp, nested 256/32)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. tab-set::

   .. tab-item:: Specialized proto

      .. code-block:: text

         QuantizationProto {
             tiling: TilingQuantizationProto {
                 tile_shape: [256]
                 elem_quant: [
                     QuantizationProto {
                         tiling: TilingQuantizationProto {
                             tile_shape: [32]
                             elem_quant: [
                                 QuantizationProto { linear: LinearUniformProto {
                                     storage_type: UINT4, bits: 3,
                                     symmetric: false, scale_float: 0.004, zero_point: 3, axis: -1
                                 }},
                                 // ... 8 sub-blocks per super-block
                             ]
                         }
                     },
                     // ... one per super-block
                 ]
             }
         }

   .. tab-item:: Custom types

      .. code-block:: text

         ModelProto {
             struct_types: [
                 StructTypeProto {
                     name: "Q3_K_SUBBLOCK_32"
                     structure: Structure {
                         field: { name: "values", type: array(UINT8, dimension=12) }
                         field: {
                             name: "storage_type"
                             constant: tensor(INT32, [], UINT4)
                         }
                         field: { name: "bits", constant: tensor(INT32, [], 3) }
                         field: { name: "symmetric", constant: tensor(BOOL, [], false) }
                         field: { name: "scale", constant: tensor(FLOAT, [], 0.004) }
                         field: { name: "zero_point", constant: tensor(INT64, [], 3) }
                         field: { name: "axis", constant: tensor(INT32, [], -1) }
                     }
                 },
                 StructTypeProto {
                     name: "Q3_K_SUPERBLOCK_256"
                     structure: Structure {
                         field: {
                             name: "subblocks"
                             type: array(struct(type_index=0), dimension=8)
                         }
                         field: {
                             name: "tile_shape"
                             constant: tensor(INT64, [1], [32])
                         }
                     }
                 },
                 StructTypeProto {
                     name: "Q3_K_TENSOR"
                     structure: Structure {
                         field: {
                             name: "superblocks"
                             type: array(
                                 struct(type_index=1),
                                 dimension=superblock_count
                             )
                         }
                         field: {
                             name: "tile_shape"
                             constant: tensor(INT64, [1], [256])
                         }
                     }
                 }
             ]
         }

Q4_K (llama.cpp, nested 256/32)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. tab-set::

   .. tab-item:: Specialized proto

      .. code-block:: text

         QuantizationProto {
             tiling: TilingQuantizationProto {
                 tile_shape: [256]
                 elem_quant: [
                     QuantizationProto {
                         tiling: TilingQuantizationProto {
                             tile_shape: [32]
                             elem_quant: [
                                 QuantizationProto { linear: LinearUniformProto {
                                     storage_type: UINT4, bits: 4,
                                     symmetric: false, scale_float: 0.003, zero_point: 2, axis: -1
                                 }},
                                 // ... 8 sub-blocks per super-block
                             ]
                         }
                     },
                     // ... one per super-block
                 ]
             }
         }

   .. tab-item:: Custom types

      .. code-block:: text

         ModelProto {
             struct_types: [
                 StructTypeProto {
                     name: "Q4_K_SUBBLOCK_32"
                     structure: Structure {
                         field: { name: "values", type: array(UINT4, dimension=32) }
                         field: {
                             name: "storage_type"
                             constant: tensor(INT32, [], UINT4)
                         }
                         field: { name: "bits", constant: tensor(INT32, [], 4) }
                         field: { name: "symmetric", constant: tensor(BOOL, [], false) }
                         field: { name: "scale", constant: tensor(FLOAT, [], 0.003) }
                         field: { name: "zero_point", constant: tensor(INT64, [], 2) }
                         field: { name: "axis", constant: tensor(INT32, [], -1) }
                     }
                 },
                 StructTypeProto {
                     name: "Q4_K_SUPERBLOCK_256"
                     structure: Structure {
                         field: {
                             name: "subblocks"
                             type: array(struct(type_index=0), dimension=8)
                         }
                         field: {
                             name: "tile_shape"
                             constant: tensor(INT64, [1], [32])
                         }
                     }
                 },
                 StructTypeProto {
                     name: "Q4_K_TENSOR"
                     structure: Structure {
                         field: {
                             name: "superblocks"
                             type: array(
                                 struct(type_index=1),
                                 dimension=superblock_count
                             )
                         }
                         field: {
                             name: "tile_shape"
                             constant: tensor(INT64, [1], [256])
                         }
                     }
                 }
             ]
         }

Q5_K (llama.cpp, nested 256/32)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. tab-set::

   .. tab-item:: Specialized proto

      .. code-block:: text

         QuantizationProto {
             tiling: TilingQuantizationProto {
                 tile_shape: [256]
                 elem_quant: [
                     QuantizationProto {
                         tiling: TilingQuantizationProto {
                             tile_shape: [32]
                             elem_quant: [
                                 QuantizationProto { linear: LinearUniformProto {
                                     storage_type: UINT8, bits: 5,
                                     symmetric: false, scale_float: 0.002, zero_point: 4, axis: -1
                                 }},
                                 // ... 8 sub-blocks per super-block
                             ]
                         }
                     },
                     // ... one per super-block
                 ]
             }
         }

   .. tab-item:: Custom types

      .. code-block:: text

         ModelProto {
             struct_types: [
                 StructTypeProto {
                     name: "Q5_K_SUBBLOCK_32"
                     structure: Structure {
                         field: { name: "values", type: array(UINT8, dimension=20) }
                         field: {
                             name: "storage_type"
                             constant: tensor(INT32, [], UINT8)
                         }
                         field: { name: "bits", constant: tensor(INT32, [], 5) }
                         field: { name: "symmetric", constant: tensor(BOOL, [], false) }
                         field: { name: "scale", constant: tensor(FLOAT, [], 0.002) }
                         field: { name: "zero_point", constant: tensor(INT64, [], 4) }
                         field: { name: "axis", constant: tensor(INT32, [], -1) }
                     }
                 },
                 StructTypeProto {
                     name: "Q5_K_SUPERBLOCK_256"
                     structure: Structure {
                         field: {
                             name: "subblocks"
                             type: array(struct(type_index=0), dimension=8)
                         }
                         field: {
                             name: "tile_shape"
                             constant: tensor(INT64, [1], [32])
                         }
                     }
                 },
                 StructTypeProto {
                     name: "Q5_K_TENSOR"
                     structure: Structure {
                         field: {
                             name: "superblocks"
                             type: array(
                                 struct(type_index=1),
                                 dimension=superblock_count
                             )
                         }
                         field: {
                             name: "tile_shape"
                             constant: tensor(INT64, [1], [256])
                         }
                     }
                 }
             ]
         }

Q6_K (llama.cpp, nested 256/16)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. tab-set::

   .. tab-item:: Specialized proto

      .. code-block:: text

         QuantizationProto {
             tiling: TilingQuantizationProto {
                 tile_shape: [256]
                 elem_quant: [
                     QuantizationProto {
                         tiling: TilingQuantizationProto {
                             tile_shape: [16]
                             elem_quant: [
                                 QuantizationProto { linear: LinearUniformProto {
                                     storage_type: UINT8, bits: 6,
                                     symmetric: false, scale_float: 0.001, zero_point: 5, axis: -1
                                 }},
                                 // ... 16 sub-blocks per super-block
                             ]
                         }
                     },
                     // ... one per super-block
                 ]
             }
         }

   .. tab-item:: Custom types

      .. code-block:: text

         ModelProto {
             struct_types: [
                 StructTypeProto {
                     name: "Q6_K_SUBBLOCK_16"
                     structure: Structure {
                         field: { name: "values", type: array(UINT8, dimension=12) }
                         field: {
                             name: "storage_type"
                             constant: tensor(INT32, [], UINT8)
                         }
                         field: { name: "bits", constant: tensor(INT32, [], 6) }
                         field: { name: "symmetric", constant: tensor(BOOL, [], false) }
                         field: { name: "scale", constant: tensor(FLOAT, [], 0.001) }
                         field: { name: "zero_point", constant: tensor(INT64, [], 5) }
                         field: { name: "axis", constant: tensor(INT32, [], -1) }
                     }
                 },
                 StructTypeProto {
                     name: "Q6_K_SUPERBLOCK_256"
                     structure: Structure {
                         field: {
                             name: "subblocks"
                             type: array(struct(type_index=0), dimension=16)
                         }
                         field: {
                             name: "tile_shape"
                             constant: tensor(INT64, [1], [16])
                         }
                     }
                 },
                 StructTypeProto {
                     name: "Q6_K_TENSOR"
                     structure: Structure {
                         field: {
                             name: "superblocks"
                             type: array(
                                 struct(type_index=1),
                                 dimension=superblock_count
                             )
                         }
                         field: {
                             name: "tile_shape"
                             constant: tensor(INT64, [1], [256])
                         }
                     }
                 }
             ]
         }

MXFP4 (OCP Microscaling, shared exponent per group of 32)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. tab-set::

   .. tab-item:: Specialized proto

      .. code-block:: text

         QuantizationProto {
             tiling: TilingQuantizationProto {
                 tile_shape: [32]
                 elem_quant: [
                     QuantizationProto { floating_point: FloatingPointUniformProto {
                         sign_bits: 1, exponent_bits: 2, mantissa_bits: 1,
                         exponent_bias: 1, has_inf: false, has_nan: false,
                         split_storage: false, packed_count: 2, packed_bytes: 1
                     }},
                     // ... one per block, shared exponent at super-block level
                 ]
             }
         }

   .. tab-item:: Custom types

      .. code-block:: text

         ModelProto {
             struct_types: [
                 StructTypeProto {
                     name: "MXFP4_E2M1_BLOCK_32"
                     structure: Structure {
                         field: {
                             name: "values"
                             type: array(FLOAT4E2M1, dimension=32)
                         }
                         field: {
                             name: "scale"
                             type: array(FLOAT8E8M0, dimension=1)
                         }
                         field: { name: "sign_bits", constant: tensor(INT32, [], 1) }
                         field: { name: "exponent_bits", constant: tensor(INT32, [], 2) }
                         field: { name: "mantissa_bits", constant: tensor(INT32, [], 1) }
                         field: { name: "exponent_bias", constant: tensor(INT32, [], 1) }
                         field: { name: "has_inf", constant: tensor(BOOL, [], false) }
                         field: { name: "has_nan", constant: tensor(BOOL, [], false) }
                         field: { name: "split_storage", constant: tensor(BOOL, [], false) }
                         field: { name: "packed_count", constant: tensor(INT32, [], 2) }
                         field: { name: "packed_bytes", constant: tensor(INT32, [], 1) }
                     }
                 },
                 StructTypeProto {
                     name: "MXFP4_TILES"
                     structure: Structure {
                         field: {
                             name: "blocks"
                             type: array(
                                 struct(type_index=0),
                                 dimension=block_count
                             )
                         }
                         field: {
                             name: "tile_shape"
                             constant: tensor(INT64, [1], [32])
                         }
                     }
                 }
             ]
         }

MXFP6 E3M2 (OCP Microscaling, 6-bit float per group of 32)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. tab-set::

   .. tab-item:: Specialized proto

      .. code-block:: text

         QuantizationProto {
             tiling: TilingQuantizationProto {
                 tile_shape: [32]
                 elem_quant: [
                     QuantizationProto { floating_point: FloatingPointUniformProto {
                         sign_bits: 1, exponent_bits: 3, mantissa_bits: 2,
                         exponent_bias: 3, has_inf: false, has_nan: false,
                         split_storage: false, packed_count: 4, packed_bytes: 3
                     }},
                     // ... one per block
                 ]
             }
         }

   .. tab-item:: Custom types

      .. code-block:: text

         ModelProto {
             struct_types: [
                 StructTypeProto {
                     name: "MXFP6_E3M2_BLOCK_32"
                     structure: Structure {
                         field: { name: "values", type: array(UINT8, dimension=24) }
                         field: {
                             name: "scale"
                             type: array(FLOAT8E8M0, dimension=1)
                         }
                         field: { name: "sign_bits", constant: tensor(INT32, [], 1) }
                         field: { name: "exponent_bits", constant: tensor(INT32, [], 3) }
                         field: { name: "mantissa_bits", constant: tensor(INT32, [], 2) }
                         field: { name: "exponent_bias", constant: tensor(INT32, [], 3) }
                         field: { name: "has_inf", constant: tensor(BOOL, [], false) }
                         field: { name: "has_nan", constant: tensor(BOOL, [], false) }
                         field: { name: "split_storage", constant: tensor(BOOL, [], false) }
                         field: { name: "packed_count", constant: tensor(INT32, [], 4) }
                         field: { name: "packed_bytes", constant: tensor(INT32, [], 3) }
                     }
                 },
                 StructTypeProto {
                     name: "MXFP6_E3M2_TILES"
                     structure: Structure {
                         field: {
                             name: "blocks"
                             type: array(
                                 struct(type_index=0),
                                 dimension=block_count
                             )
                         }
                         field: {
                             name: "tile_shape"
                             constant: tensor(INT64, [1], [32])
                         }
                     }
                 }
             ]
         }

FP6 LLM (DeepSpeed TC-FPn, per-group of 128)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Same FP6 E3M2 quantization as MXFP6 but with split storage
(4-bit sign+exponent and 2-bit mantissa stored separately) for
Tensor Core alignment.

.. tab-set::

   .. tab-item:: Specialized proto

      .. code-block:: text

         QuantizationProto {
             tiling: TilingQuantizationProto {
                 tile_shape: [128]
                 elem_quant: [
                     QuantizationProto {
                         floating_point: FloatingPointUniformProto {
                             sign_bits: 1
                             exponent_bits: 3
                             mantissa_bits: 2
                             exponent_bias: 3
                             has_inf: false
                             has_nan: false
                             split_storage: true
                             packed_count: 4
                             packed_bytes: 3
                         }
                     }
                 ]
             }
         }

   .. tab-item:: Custom types

      .. code-block:: text

         ModelProto {
             struct_types: [
                 StructTypeProto {  // index 0: four FLOAT6_E3M2 values
                     name: "FLOAT6_E3M2"
                     structure: Structure {
                         field: {
                             name: "sign_exponent"
                             type: bit_packing(
                                 dimension=4,
                                 components=[sign:1, exponent:3]
                             )
                         }
                         field: {
                             name: "mantissa"
                             type: array(UINT2, dimension=4)
                         }
                         field: { name: "exponent_bias", constant: tensor(INT32, [], 3) }
                         field: { name: "has_inf", constant: tensor(BOOL, [], false) }
                         field: { name: "has_nan", constant: tensor(BOOL, [], false) }
                     }
                 },
                 StructTypeProto {  // index 1: one tile
                     name: "FP6_LLM_TILE_128"
                     structure: Structure {
                         field: {
                             name: "values"
                             type: array(
                                 struct(type_index=0),
                                 dimension=32
                             )
                         }
                         field: {
                             name: "tile_shape"
                             constant: tensor(INT64, [1], [128])
                         }
                     }
                 }
             ]
         }

INT8 per-channel (classic CNN)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. tab-set::

   .. tab-item:: Specialized proto

      .. code-block:: text

         QuantizationProto {
             linear: LinearUniformProto {
                 storage_type: INT8, bits: 8, symmetric: true,
                 scale_float: 0.03, zero_point: 0, axis: 0
             }
         }

   .. tab-item:: Custom types

      .. code-block:: text

         StructTypeProto {
             name: "INT8_PER_CHANNEL"
             structure: Structure {
                 field: { name: "values", type: array(INT8, dimension=N) }
                 field: {
                     name: "storage_type"
                     constant: tensor(INT32, [], INT8)
                 }
                 field: { name: "bits", constant: tensor(INT32, [], 8) }
                 field: { name: "symmetric", constant: tensor(BOOL, [], true) }
                 field: { name: "scale", constant: tensor(FLOAT, [], 0.03) }
                 field: { name: "zero_point", constant: tensor(INT64, [], 0) }
                 field: { name: "axis", constant: tensor(INT32, [], 0) }
             }
         }

1.58-bit ternary (BitNet b1.58)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. tab-set::

   .. tab-item:: Specialized proto

      .. code-block:: text

         QuantizationProto {
             codebook: CodebookProto {
                 scale_float: 0.5,
                 codebook_data: [-1.0, 0.0, 1.0],
                 packed_count: 5, packed_bytes: 1
             }
         }

   .. tab-item:: Custom types

      .. code-block:: text

         StructTypeProto {
             name: "TERNARY_1_58"
             structure: Structure {
                 field: {
                     name: "indices"
                     type: array(UINT8, dimension=packed_byte_count)
                 }
                 field: {
                     name: "codebook"
                     constant: tensor(FLOAT, [3], [-1.0, 0.0, 1.0])
                 }
                 field: { name: "scale", constant: tensor(FLOAT, [], 0.5) }
                 field: { name: "packed_count", constant: tensor(INT32, [], 5) }
                 field: { name: "packed_bytes", constant: tensor(INT32, [], 1) }
                 field: { name: "num_codebooks", constant: tensor(INT32, [], 1) }
                 field: { name: "codebook_size", constant: tensor(INT32, [], 3) }
                 field: { name: "vector_size", constant: tensor(INT32, [], 1) }
                 field: { name: "index_bits", constant: tensor(INT32, [], 0) }
             }
         }

FP8 E4M3 (per-tensor)
^^^^^^^^^^^^^^^^^^^^^^

.. tab-set::

   .. tab-item:: Specialized proto

      .. code-block:: text

         QuantizationProto {
             linear: LinearUniformProto {
                 storage_type: FLOAT8E4M3FN, bits: 8, symmetric: true,
                 scale_float: 1.0, zero_point: 0, axis: -1
             }
         }

   .. tab-item:: Custom types

      .. code-block:: text

         StructTypeProto {
             name: "FP8_E4M3_PER_TENSOR"
             structure: Structure {
                 field: {
                     name: "values"
                     type: array(FLOAT8E4M3FN, dimension=N)
                 }
                 field: {
                     name: "storage_type"
                     constant: tensor(INT32, [], FLOAT8E4M3FN)
                 }
                 field: { name: "bits", constant: tensor(INT32, [], 8) }
                 field: { name: "symmetric", constant: tensor(BOOL, [], true) }
                 field: { name: "scale", constant: tensor(FLOAT, [], 1.0) }
                 field: { name: "zero_point", constant: tensor(INT64, [], 0) }
                 field: { name: "axis", constant: tensor(INT32, [], -1) }
             }
         }

AQLM 2×8 (Additive Quantization, 2 bits/weight)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. tab-set::

   .. tab-item:: Specialized proto

      .. code-block:: text

         QuantizationProto {
             codebook: CodebookProto {
                 num_codebooks: 2,
                 codebook_size: 256,
                 vector_size: 8,
                 index_bits: 8,
                 codebook_data: [...]  // 2 * 256 * 8 = 4096 floats
             }
         }

   .. tab-item:: Custom types

      .. code-block:: text

         StructTypeProto {
             name: "AQLM_2X8"
             structure: Structure {
                 field: {
                     name: "indices"
                     type: array(UINT8, dimensions=[vector_count, 2])
                 }
                 field: {
                     name: "codebooks"
                     constant: tensor(
                         FLOAT,
                         [2, 256, 8],
                         aqlm_codebook_data
                     )
                 }
                 field: { name: "num_codebooks", constant: tensor(INT32, [], 2) }
                 field: { name: "codebook_size", constant: tensor(INT32, [], 256) }
                 field: { name: "vector_size", constant: tensor(INT32, [], 8) }
                 field: { name: "index_bits", constant: tensor(INT32, [], 8) }
             }
         }

IQ1_S (llama.cpp, 1.56 bits/weight, E8 lattice)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Uses a 2048-entry vector codebook (E8 lattice grid, 8 values per entry).
Each sub-block of 8 weights is looked up with an 11-bit index.
Wrapped in a block of 256 for the shared FP16 scale.

.. tab-set::

   .. tab-item:: Specialized proto

      .. code-block:: text

         QuantizationProto {
             tiling: TilingQuantizationProto {
                 tile_shape: [256]
                 elem_quant: [
                     QuantizationProto { codebook: CodebookProto {
                         num_codebooks: 1,
                         codebook_size: 2048,
                         vector_size: 8,
                         index_bits: 11,
                         codebook_data: [...]  // 2048 * 8 = 16384 values in {-1, 0, 1}
                     }},
                     // ... one per super-block
                 ]
             }
         }

   .. tab-item:: Custom types

      .. code-block:: text

         ModelProto {
             struct_types: [
                 StructTypeProto {
                     name: "IQ1_S_BLOCK_256"
                     structure: Structure {
                         field: {
                             name: "indices"
                             type: array(UINT8, dimension=44)
                         }
                         field: {
                             name: "scale"
                             type: array(FLOAT16, dimension=1)
                         }
                         field: {
                             name: "codebook"
                             constant: tensor(
                                 FLOAT,
                                 [2048, 8],
                                 iq1_s_e8_codebook
                             )
                         }
                         field: {
                             name: "num_codebooks"
                             constant: tensor(INT32, [], 1)
                         }
                         field: {
                             name: "codebook_size"
                             constant: tensor(INT32, [], 2048)
                         }
                         field: {
                             name: "vector_size"
                             constant: tensor(INT32, [], 8)
                         }
                         field: {
                             name: "index_bits"
                             constant: tensor(INT32, [], 11)
                         }
                     }
                 },
                 StructTypeProto {
                     name: "IQ1_S_TILES"
                     structure: Structure {
                         field: {
                             name: "blocks"
                             type: array(
                                 struct(type_index=0),
                                 dimension=block_count
                             )
                         }
                         field: {
                             name: "tile_shape"
                             constant: tensor(INT64, [1], [256])
                         }
                     }
                 }
             ]
         }

SpQR (Sparse Quantization, ~3.4 bits/weight)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Outliers (>1% of values) stored in FP16, rest in INT3 per-group.

.. tab-set::

   .. tab-item:: Specialized proto

      .. code-block:: text

         QuantizationProto {
             sparse: SparseQuantizationProto {
                 base_quant: QuantizationProto {
                     tiling: TilingQuantizationProto {
                         tile_shape: [16]
                         elem_quant: [
                             QuantizationProto { linear: LinearUniformProto {
                                 storage_type: INT4, bits: 3,
                                 symmetric: false, scale_float: 0.01, zero_point: 4, axis: -1
                             }},
                             // ... one per block
                         ]
                     }
                 },
                 outlier_data_type: FLOAT16,
                 outlier_threshold: 6.0,
                 outlier_ratio: 0.01
             }
         }

   .. tab-item:: Custom types

      .. code-block:: text

         ModelProto {
             struct_types: [
                 StructTypeProto {
                     name: "SPQR_INT3_BLOCK_16"
                     structure: Structure {
                         field: { name: "values", type: array(UINT8, dimension=6) }
                         field: {
                             name: "storage_type"
                             constant: tensor(INT32, [], INT4)
                         }
                         field: { name: "bits", constant: tensor(INT32, [], 3) }
                         field: { name: "symmetric", constant: tensor(BOOL, [], false) }
                         field: { name: "scale", constant: tensor(FLOAT, [], 0.01) }
                         field: { name: "zero_point", constant: tensor(INT64, [], 4) }
                         field: { name: "axis", constant: tensor(INT32, [], -1) }
                     }
                 },
                 StructTypeProto {
                     name: "SPQR_DENSE_TILES"
                     structure: Structure {
                         field: {
                             name: "blocks"
                             type: array(
                                 struct(type_index=0),
                                 dimension=dense_block_count
                             )
                         }
                         field: {
                             name: "tile_shape"
                             constant: tensor(INT64, [1], [16])
                         }
                     }
                 },
                 StructTypeProto {
                     name: "SPQR"
                     structure: Structure {
                         field: {
                             name: "base"
                             type: struct(type_index=1)
                         }
                         field: {
                             name: "outlier_indices"
                             type: array(INT64, dimension=outlier_count)
                         }
                         field: {
                             name: "outlier_values"
                             type: array(FLOAT16, dimension=outlier_count)
                         }
                         field: {
                             name: "outlier_data_type"
                             constant: tensor(INT32, [], FLOAT16)
                         }
                         field: {
                             name: "outlier_threshold"
                             constant: tensor(FLOAT, [], 6.0)
                         }
                         field: {
                             name: "outlier_ratio"
                             constant: tensor(FLOAT, [], 0.01)
                         }
                     }
                 }
             ]
         }

QuIP# (Vector quantization with Hadamard rotation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. tab-set::

   .. tab-item:: Specialized proto

      .. code-block:: text

         QuantizationProto {
             codebook: CodebookProto {
                 num_codebooks: 1,
                 codebook_size: 256,
                 vector_size: 8,
                 index_bits: 8,
                 codebook_data: [...]  // learned codebook, 256 * 8 = 2048 floats
             },
             pre_rotation: RotationProto { matrix_type: HADAMARD, dims: [4096, 4096] },
             post_rotation: RotationProto { matrix_type: HADAMARD, dims: [4096, 4096] }
         }

   .. tab-item:: Custom types

      .. code-block:: text

         ModelProto {
             struct_types: [
                 StructTypeProto {
                     name: "QUIP_SHARP"
                     structure: Structure {
                         field: {
                             name: "indices"
                             type: array(UINT8, dimension=vector_count)
                         }
                         field: {
                             name: "codebook"
                             constant: tensor(
                                 FLOAT,
                                 [256, 8],
                                 quip_sharp_codebook
                             )
                         }
                         field: {
                             name: "num_codebooks"
                             constant: tensor(INT32, [], 1)
                         }
                         field: {
                             name: "codebook_size"
                             constant: tensor(INT32, [], 256)
                         }
                         field: {
                             name: "vector_size"
                             constant: tensor(INT32, [], 8)
                         }
                         field: {
                             name: "index_bits"
                             constant: tensor(INT32, [], 8)
                         }
                         field: {
                             name: "pre_rotation_type"
                             constant: tensor(INT32, [], 0)  // HADAMARD
                         }
                         field: {
                             name: "pre_rotation_dims"
                             constant: tensor(INT32, [2], [4096, 4096])
                         }
                         field: {
                             name: "post_rotation_type"
                             constant: tensor(INT32, [], 0)  // HADAMARD
                         }
                         field: {
                             name: "post_rotation_dims"
                             constant: tensor(INT32, [2], [4096, 4096])
                         }
                     }
                 }
             ]
         }

EXL2 (variable bits per layer, ~3.5 bpw average)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

EXL2 assigns different bit-widths per layer to hit a target average bpw.
Each layer gets its own ``QuantizationProto`` (via ``quantized_type`` index),
so a model may use several quantization entries — e.g. 2-bit for less
important layers and 4-bit for critical ones.

.. tab-set::

   .. tab-item:: Specialized proto

      .. code-block:: text

         // Layer A (less important): 2-bit per group of 128
         QuantizationProto {
             tiling: TilingQuantizationProto {
                 tile_shape: [128]
                 elem_quant: [
                     QuantizationProto { linear: LinearUniformProto {
                         storage_type: INT4, bits: 2,
                         symmetric: false, scale_float: 0.005, zero_point: 2, axis: -1
                     }},
                     // ... one per block
                 ]
             }
         }

         // Layer B (critical): 4-bit per group of 128
         QuantizationProto {
             tiling: TilingQuantizationProto {
                 tile_shape: [128]
                 elem_quant: [
                     QuantizationProto { linear: LinearUniformProto {
                         storage_type: INT4, bits: 4,
                         symmetric: false, scale_float: 0.01, zero_point: 8, axis: -1
                     }},
                     // ... one per block
                 ]
             }
         }

   .. tab-item:: Custom types

      .. code-block:: text

         ModelProto {
             struct_types: [
                 StructTypeProto {
                     name: "EXL2_INT2_BLOCK_128"
                     structure: Structure {
                         field: { name: "values", type: array(UINT8, dimension=32) }
                         field: {
                             name: "storage_type"
                             constant: tensor(INT32, [], INT4)
                         }
                         field: { name: "bits", constant: tensor(INT32, [], 2) }
                         field: { name: "symmetric", constant: tensor(BOOL, [], false) }
                         field: { name: "scale", constant: tensor(FLOAT, [], 0.005) }
                         field: { name: "zero_point", constant: tensor(INT64, [], 2) }
                         field: { name: "axis", constant: tensor(INT32, [], -1) }
                     }
                 },
                 StructTypeProto {
                     name: "EXL2_INT4_BLOCK_128"
                     structure: Structure {
                         field: { name: "values", type: array(INT4, dimension=128) }
                         field: {
                             name: "storage_type"
                             constant: tensor(INT32, [], INT4)
                         }
                         field: { name: "bits", constant: tensor(INT32, [], 4) }
                         field: { name: "symmetric", constant: tensor(BOOL, [], false) }
                         field: { name: "scale", constant: tensor(FLOAT, [], 0.01) }
                         field: { name: "zero_point", constant: tensor(INT64, [], 8) }
                         field: { name: "axis", constant: tensor(INT32, [], -1) }
                     }
                 },
                 StructTypeProto {
                     name: "EXL2_LAYER_A"
                     structure: Structure {
                         field: {
                             name: "blocks"
                             type: array(
                                 struct(type_index=0),
                                 dimension=layer_a_block_count
                             )
                         }
                         field: {
                             name: "tile_shape"
                             constant: tensor(INT64, [1], [128])
                         }
                     }
                 },
                 StructTypeProto {
                     name: "EXL2_LAYER_B"
                     structure: Structure {
                         field: {
                             name: "blocks"
                             type: array(
                                 struct(type_index=1),
                                 dimension=layer_b_block_count
                             )
                         }
                         field: {
                             name: "tile_shape"
                             constant: tensor(INT64, [1], [128])
                         }
                     }
                 }
             ]
         }

Log quantization (4-bit)
^^^^^^^^^^^^^^^^^^^^^^^^^

.. tab-set::

   .. tab-item:: Specialized proto

      .. code-block:: text

         QuantizationProto {
             log: LogUniformProto {
                 bits: 4,
                 base: 2.0,
                 offset: -7.0,
                 has_sign: true
             }
         }

   .. tab-item:: Custom types

      .. code-block:: text

         StructTypeProto {
             name: "LOG4"
             structure: Structure {
                 field: { name: "codes", type: array(UINT4, dimension=N) }
                 field: {
                     name: "signs"
                     type: array(UINT8, dimension=sign_byte_count)
                 }
                 field: { name: "bits", constant: tensor(INT32, [], 4) }
                 field: { name: "base", constant: tensor(FLOAT, [], 2.0) }
                 field: { name: "offset", constant: tensor(FLOAT, [], -7.0) }
                 field: { name: "has_sign", constant: tensor(BOOL, [], true) }
             }
         }

Custom (plugin-based)
^^^^^^^^^^^^^^^^^^^^^

.. tab-set::

   .. tab-item:: Specialized proto

      .. code-block:: text

         QuantizationProto {
             function: FunctionUniformProto {
                 storage_type: UINT4, bits: 4,
                 quantize_op: "vendor::QuantizeV2",
                 dequantize_op: "vendor::DequantizeV2"
             }
         }

   .. tab-item:: Custom types

      .. code-block:: text

         StructTypeProto {
             name: "VENDOR_QUANTIZE_V2"
             structure: Structure {
                 field: { name: "payload", type: array(UINT4, dimension=N) }
                 field: {
                     name: "storage_type"
                     constant: tensor(INT32, [], UINT4)
                 }
                 field: { name: "bits", constant: tensor(INT32, [], 4) }
             }
             metadata_props: {
                 key: "quantize_op"
                 value: "vendor::QuantizeV2"
             }
             metadata_props: {
                 key: "dequantize_op"
                 value: "vendor::DequantizeV2"
             }
         }

MatMulNBits INT4 (onnxruntime, per-group of 32, tiled)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Weights of shape ``[K, N]`` are tiled into blocks of 32 along axis K
and 128 along axis N (SIMD width). All tiles share the same INT4
quantization scheme. ``perm: [1, 0]`` indicates N-major tile ordering
in memory for cache locality.

.. tab-set::

   .. tab-item:: Specialized proto

      .. code-block:: text

         // Weight shape: [4096, 4096], group_size=32, N_tile=128
         QuantizationProto {
             tiling: TilingQuantizationProto {
                 tile_shape: [32, 128],
                 axes: [0, 1],
                 elem_quant: [QuantizationProto { linear: LinearUniformProto {
                     storage_type: UINT4, bits: 4,
                     symmetric: true, scale_float: 0.0, zero_point: 0, axis: -1
                 }}],
                 perm: [1, 0]   // N-major ordering in memory
             }
         }

   .. tab-item:: Custom types

      .. code-block:: text

         ModelProto {
             struct_types: [
                 StructTypeProto {
                     name: "MATMUL_NBITS_INT4_TILE_32X128"
                     structure: Structure {
                         field: {
                             name: "values"
                             type: array(UINT4, dimensions=[32, 128])
                         }
                         field: {
                             name: "storage_type"
                             constant: tensor(INT32, [], UINT4)
                         }
                         field: { name: "bits", constant: tensor(INT32, [], 4) }
                         field: { name: "symmetric", constant: tensor(BOOL, [], true) }
                         field: { name: "scale", constant: tensor(FLOAT, [], 0.0) }
                         field: { name: "zero_point", constant: tensor(INT64, [], 0) }
                         field: { name: "axis", constant: tensor(INT32, [], -1) }
                     }
                 },
                 StructTypeProto {
                     name: "MATMUL_NBITS_INT4_4096X4096"
                     structure: Structure {
                         field: {
                             name: "tiles"
                             type: array(
                                 struct(type_index=0),
                                 dimensions=[32, 128]
                             )
                         }
                         field: {
                             name: "tile_shape"
                             constant: tensor(INT64, [2], [32, 128])
                         }
                         field: {
                             name: "axes"
                             constant: tensor(INT32, [2], [0, 1])
                         }
                         field: {
                             name: "perm"
                             constant: tensor(INT32, [2], [1, 0])
                         }
                     }
                 }
             ]
         }

Block-tiled floats (no quantization)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Reorganizes a row-major float matrix into tiles of ``[32, 128]``
without any value transformation. Useful for prepack layouts
that expect block-tiled storage.

.. tab-set::

   .. tab-item:: Specialized proto

      .. code-block:: text

         // Weight shape: [4096, 4096], tiled into [32, 128] blocks
         QuantizationProto {
             tiling: TilingQuantizationProto {
                 tile_shape: [32, 128],
                 axes: [0, 1],
                 elem_quant: [QuantizationProto {
                     data_type: FLOAT,
                     cast: CastUniformProto { storage_type: FLOAT }
                 }]
             }
         }

   .. tab-item:: Custom types

      .. code-block:: text

         ModelProto {
             struct_types: [
                 StructTypeProto {
                     name: "FLOAT_TILE_32X128"
                     array: Array {
                         element_type: FLOAT
                         dimension: 32
                         dimension: 128
                     }
                 },
                 StructTypeProto {
                     name: "BLOCK_TILED_FLOAT_4096X4096"
                     structure: Structure {
                         field: {
                             name: "tiles"
                             type: array(
                                 struct(type_index=0),
                                 dimensions=[128, 32]
                             )
                         }
                         field: {
                             name: "tile_shape"
                             constant: tensor(INT64, [2], [32, 128])
                         }
                         field: {
                             name: "axes"
                             constant: tensor(INT32, [2], [0, 1])
                         }
                     }
                 }
             ]
         }

STQ1_0 (Sherry, 1.25 bits/weight, ternary codebook)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

256 values per 42-byte block. Each 5-bit combined index (4-bit code +
1-bit sign) selects from a 32-entry codebook of 4 ternary values.
Output positions use a stride-16 scatter within each 64-value chunk.

.. tab-set::

   .. tab-item:: Specialized proto

      .. code-block:: text

         QuantizationProto {
             data_type: FLOAT,
             structured_block: StructuredBlockUniformProto {
                 block_layout: BlockLayoutProto {
                     block_size: 256, bytes_per_block: 42,
                     fields: [
                         BlockFieldProto { role: CODE,  bit_offset: 0,   bit_width: 4,  count: 64, data_type: 0 },
                         BlockFieldProto { role: SIGN,  bit_offset: 256, bit_width: 1,  count: 64, data_type: 0 },
                         BlockFieldProto { role: SCALE, bit_offset: 320, bit_width: 16, count: 1,  data_type: FLOAT16 },
                     ]
                 },
                 codebook_data: [...]  // 32 * 4 = 128 ternary values in {-1, 0, 1}
                 codebook_vector_size: 4,
                 index_formula: [
                     FieldWeightProto { field: CODE, multiplier: 1 },
                     FieldWeightProto { field: SIGN, multiplier: 16 },
                 ],
                 scatter: ScatterProto { group_size: 64, vector_size: 4, stride: 16 }
             }
         }

   .. tab-item:: Custom types

      .. code-block:: text

         ModelProto {
             struct_types: [
                 StructTypeProto {
                     name: "STQ1_0_BLOCK_256"
                     structure: Structure {
                         field: { name: "code", type: array(UINT4, dimension=64) }
                         field: { name: "sign", type: array(UINT8, dimension=8) }
                         field: { name: "scale", type: array(FLOAT16, dimension=1) }
                         field: {
                             name: "field_roles"
                             constant: tensor(INT32, [3], [6, 1, 2])
                         }
                         field: {
                             name: "sign_bit_width"
                             constant: tensor(INT32, [], 1)
                         }
                         field: {
                             name: "sign_count"
                             constant: tensor(INT32, [], 64)
                         }
                         field: {
                             name: "block_size"
                             constant: tensor(INT32, [], 256)
                         }
                     }
                 },
                 StructTypeProto {
                     name: "STQ1_0"
                     structure: Structure {
                         field: {
                             name: "blocks"
                             type: array(
                                 struct(type_index=0),
                                 dimension=block_count
                             )
                         }
                         field: {
                             name: "codebook"
                             constant: tensor(
                                 FLOAT,
                                 [32, 4],
                                 stq1_0_codebook
                             )
                         }
                         field: {
                             name: "codebook_vector_size"
                             constant: tensor(INT32, [], 4)
                         }
                         field: {
                             name: "index_fields"
                             constant: tensor(INT32, [2], [6, 1])
                         }
                         field: {
                             name: "index_multipliers"
                             constant: tensor(INT32, [2], [1, 16])
                         }
                         field: {
                             name: "scatter_group_size"
                             constant: tensor(INT32, [], 64)
                         }
                         field: {
                             name: "scatter_vector_size"
                             constant: tensor(INT32, [], 4)
                         }
                         field: {
                             name: "scatter_stride"
                             constant: tensor(INT32, [], 16)
                         }
                         field: {
                             name: "data_type"
                             constant: tensor(INT32, [], FLOAT)
                         }
                     }
                 }
             ]
         }

Dequantization of one block (derived from proto fields):

.. code-block:: python

    sb = proto.structured_block
    bl = sb.block_layout
    code_field = bl.fields[role == CODE]    # bit_offset=0, bit_width=4, count=64
    sign_field = bl.fields[role == SIGN]    # bit_offset=256, bit_width=1, count=64
    scale_field = bl.fields[role == SCALE]  # bit_offset=320, bit_width=16, count=1

    scale = extract(block, scale_field.bit_offset, scale_field.bit_width, scale_field.data_type)

    for g in range(code_field.count):
        # Extract fields
        code = extract(block, code_field.bit_offset + code_field.bit_width * g, code_field.bit_width)
        sign = extract(block, sign_field.bit_offset + sign_field.bit_width * g, sign_field.bit_width)

        # Combine fields into codebook index using index_formula
        index = sum(values[fw.field] * fw.multiplier for fw in sb.index_formula)
        # = code * 1 + sign * 16

        # Codebook lookup
        vector = sb.codebook_data[index * sb.codebook_vector_size : (index+1) * sb.codebook_vector_size]

        # Scatter to output positions
        s = sb.scatter
        groups_per_chunk = s.group_size // s.vector_size  # 64 // 4 = 16
        chunk = g // groups_per_chunk
        lane = g % groups_per_chunk
        for p in range(s.vector_size):
            output[s.group_size * chunk + lane + s.stride * p] = vector[p] * scale

Column-major layout
^^^^^^^^^^^^^^^^^^^

Expresses a column-major (Fortran-order) storage of a 2D matrix.
A ``tile_shape`` value of 0 means "same as the tensor dimension on that axis",
so ``[0, 0]`` is a single tile covering the whole tensor.
``perm: [1, 0]`` reverses the axis order in memory.

.. tab-set::

   .. tab-item:: Specialized proto

      .. code-block:: text

         // Weight shape: [M, N], stored column-major
         QuantizationProto {
             tiling: TilingQuantizationProto {
                 tile_shape: [0, 0],
                 axes: [0, 1],
                 elem_quant: [QuantizationProto {
                     data_type: FLOAT,
                     cast: CastUniformProto { storage_type: FLOAT }
                 }],
                 perm: [1, 0]
             }
         }

   .. tab-item:: Custom types

      .. code-block:: text

         ModelProto {
             struct_types: [
                 StructTypeProto {
                     name: "FLOAT_MATRIX_MXN"
                     array: Array {
                         element_type: FLOAT
                         dimension: M
                         dimension: N
                     }
                 },
                 StructTypeProto {
                     name: "COLUMN_MAJOR_FLOAT_MXN"
                     structure: Structure {
                         field: {
                             name: "tiles"
                             type: array(
                                 struct(type_index=0),
                                 dimension=1
                             )
                         }
                         field: {
                             name: "tile_shape"
                             constant: tensor(INT64, [2], [0, 0])
                         }
                         field: {
                             name: "axes"
                             constant: tensor(INT32, [2], [0, 1])
                         }
                         field: {
                             name: "perm"
                             constant: tensor(INT32, [2], [1, 0])
                         }
                     }
                 }
             ]
         }

TQ1_0 (llama.cpp, 1.6 bits/weight, ternary)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pure ternary: each weight is {-1, 0, +1}. Five ternary values are packed
into one byte (3⁵ = 243 ≤ 255). One FP16 scale per block of 256 weights.

.. tab-set::

   .. tab-item:: Specialized proto

      .. code-block:: text

         // Codebook: [-1, 0, +1], packed 5 per byte
         QuantizationProto {
             data_type: FLOAT16,
             tiling: TilingQuantizationProto {
                 tile_shape: [256],
                 elem_quant: [QuantizationProto {
                     codebook: CodebookProto {
                         scale_float: 1.0,       // scale is per-block, stored separately
                         codebook_data: [-1.0, 0.0, 1.0],
                         packed_count: 5,        // 5 ternary values per byte
                         packed_bytes: 1
                     }
                 }]
             }
         }

         // Dequantize: for each block of 256 weights
         // indices = unpack_base3(data, 5 values/byte)
         // result = [codebook[i] * scale for i in indices]
         //        = [{-1,0,+1}[i] * d for i in indices]

   .. tab-item:: Custom types

      .. code-block:: text

         ModelProto {
             struct_types: [
                 StructTypeProto {
                     name: "TQ1_0_BLOCK_256"
                     structure: Structure {
                         field: { name: "indices", type: array(UINT8, dimension=52) }
                         field: { name: "scale", type: array(FLOAT16, dimension=1) }
                         field: {
                             name: "codebook"
                             constant: tensor(FLOAT, [3], [-1.0, 0.0, 1.0])
                         }
                         field: {
                             name: "packed_count"
                             constant: tensor(INT32, [], 5)
                         }
                         field: {
                             name: "packed_bytes"
                             constant: tensor(INT32, [], 1)
                         }
                     }
                 },
                 StructTypeProto {
                     name: "TQ1_0"
                     structure: Structure {
                         field: {
                             name: "blocks"
                             type: array(
                                 struct(type_index=0),
                                 dimension=block_count
                             )
                         }
                         field: {
                             name: "tile_shape"
                             constant: tensor(INT64, [1], [256])
                         }
                         field: {
                             name: "data_type"
                             constant: tensor(INT32, [], FLOAT16)
                         }
                     }
                 }
             ]
         }

TQ2_0 (llama.cpp, 2 bits/weight, ternary)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ternary with 2 bits per weight (simpler packing, 1 unused state).
Mapping: 0 → -1, 1 → 0, 2 → +1.

.. tab-set::

   .. tab-item:: Specialized proto

      .. code-block:: text

         // Codebook: [-1, 0, +1], 2 bits per value
         QuantizationProto {
             data_type: FLOAT16,
             tiling: TilingQuantizationProto {
                 tile_shape: [256],
                 elem_quant: [QuantizationProto {
                     codebook: CodebookProto {
                         scale_float: 1.0,
                         codebook_data: [-1.0, 0.0, 1.0],
                         packed_count: 4,        // 4 values per byte (2 bits each)
                         packed_bytes: 1
                     }
                 }]
             }
         }

         // Dequantize: same as TQ1_0 but simpler unpack (2 bits, not base-3)

   .. tab-item:: Custom types

      .. code-block:: text

         ModelProto {
             struct_types: [
                 StructTypeProto {
                     name: "TQ2_0_BLOCK_256"
                     structure: Structure {
                         field: { name: "indices", type: array(UINT8, dimension=64) }
                         field: { name: "scale", type: array(FLOAT16, dimension=1) }
                         field: {
                             name: "codebook"
                             constant: tensor(FLOAT, [3], [-1.0, 0.0, 1.0])
                         }
                         field: {
                             name: "packed_count"
                             constant: tensor(INT32, [], 4)
                         }
                         field: {
                             name: "packed_bytes"
                             constant: tensor(INT32, [], 1)
                         }
                     }
                 },
                 StructTypeProto {
                     name: "TQ2_0"
                     structure: Structure {
                         field: {
                             name: "blocks"
                             type: array(
                                 struct(type_index=0),
                                 dimension=block_count
                             )
                         }
                         field: {
                             name: "tile_shape"
                             constant: tensor(INT64, [1], [256])
                         }
                         field: {
                             name: "data_type"
                             constant: tensor(INT32, [], FLOAT16)
                         }
                     }
                 }
             ]
         }

BitNet b1.58 (Microsoft, 1.58 bits/weight, native ternary)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Model trained natively with ternary weights {-1, 0, +1}.
Storage is identical to TQ1_0 (5 ternary values per byte).
The difference is in training (QAT), not in the storage format.

.. tab-set::

   .. tab-item:: Specialized proto

      .. code-block:: text

         // Same storage as TQ1_0
         QuantizationProto {
             data_type: FLOAT16,
             tiling: TilingQuantizationProto {
                 tile_shape: [256],
                 elem_quant: [QuantizationProto {
                     codebook: CodebookProto {
                         scale_float: 1.0,
                         codebook_data: [-1.0, 0.0, 1.0],
                         packed_count: 5,
                         packed_bytes: 1
                     }
                 }]
             }
         }

   .. tab-item:: Custom types

      .. code-block:: text

         ModelProto {
             struct_types: [
                 StructTypeProto {
                     name: "BITNET_B1_58_BLOCK_256"
                     structure: Structure {
                         field: { name: "indices", type: array(UINT8, dimension=52) }
                         field: { name: "scale", type: array(FLOAT16, dimension=1) }
                         field: {
                             name: "codebook"
                             constant: tensor(FLOAT, [3], [-1.0, 0.0, 1.0])
                         }
                         field: {
                             name: "packed_count"
                             constant: tensor(INT32, [], 5)
                         }
                         field: {
                             name: "packed_bytes"
                             constant: tensor(INT32, [], 1)
                         }
                     }
                 },
                 StructTypeProto {
                     name: "BITNET_B1_58"
                     structure: Structure {
                         field: {
                             name: "blocks"
                             type: array(
                                 struct(type_index=0),
                                 dimension=block_count
                             )
                         }
                         field: {
                             name: "tile_shape"
                             constant: tensor(INT64, [1], [256])
                         }
                         field: {
                             name: "data_type"
                             constant: tensor(INT32, [], FLOAT16)
                         }
                     }
                 }
             ]
         }

ParetoQ (Meta, 1.58–2 bits/weight, SEQ ternary/2-bit)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Quantization-aware training with Stretched Elastic Quantization (SEQ).
The storage format is standard ternary or 2-bit; the custom SEQ function
is used only during training. At inference, the model uses a standard
codebook. If the SEQ rounding function is needed at inference, use
``FunctionUniformProto``.

.. tab-set::

   .. tab-item:: Specialized proto

      .. code-block:: text

         // Ternary variant (same storage as TQ1_0)
         QuantizationProto {
             data_type: FLOAT16,
             tiling: TilingQuantizationProto {
                 tile_shape: [256],
                 elem_quant: [QuantizationProto {
                     codebook: CodebookProto {
                         scale_float: 1.0,
                         codebook_data: [-1.0, 0.0, 1.0],
                         packed_count: 5,
                         packed_bytes: 1
                     }
                 }]
             }
         }

         // 2-bit variant (4 levels)
         QuantizationProto {
             data_type: FLOAT16,
             tiling: TilingQuantizationProto {
                 tile_shape: [128],
                 elem_quant: [QuantizationProto {
                     linear: LinearUniformProto {
                         storage_type: UINT8,
                         bits: 2,
                         symmetric: false,
                         scale_float: 0.023,
                         zero_point: 1,
                         axis: -1
                     }
                 }]
             }
         }

   .. tab-item:: Custom types

      .. code-block:: text

         ModelProto {
             struct_types: [
                 StructTypeProto {
                     name: "PARETOQ_TERNARY_BLOCK_256"
                     structure: Structure {
                         field: { name: "indices", type: array(UINT8, dimension=52) }
                         field: { name: "scale", type: array(FLOAT16, dimension=1) }
                         field: {
                             name: "codebook"
                             constant: tensor(FLOAT, [3], [-1.0, 0.0, 1.0])
                         }
                         field: {
                             name: "packed_count"
                             constant: tensor(INT32, [], 5)
                         }
                         field: {
                             name: "packed_bytes"
                             constant: tensor(INT32, [], 1)
                         }
                     }
                 },
                 StructTypeProto {
                     name: "PARETOQ_UINT2_BLOCK_128"
                     structure: Structure {
                         field: { name: "values", type: array(UINT8, dimension=32) }
                         field: {
                             name: "storage_type"
                             constant: tensor(INT32, [], UINT8)
                         }
                         field: { name: "bits", constant: tensor(INT32, [], 2) }
                         field: { name: "symmetric", constant: tensor(BOOL, [], false) }
                         field: { name: "scale", constant: tensor(FLOAT, [], 0.023) }
                         field: { name: "zero_point", constant: tensor(INT64, [], 1) }
                         field: { name: "axis", constant: tensor(INT32, [], -1) }
                     }
                 },
                 StructTypeProto {
                     name: "PARETOQ_TERNARY"
                     structure: Structure {
                         field: {
                             name: "blocks"
                             type: array(
                                 struct(type_index=0),
                                 dimension=ternary_block_count
                             )
                         }
                         field: {
                             name: "tile_shape"
                             constant: tensor(INT64, [1], [256])
                         }
                         field: {
                             name: "data_type"
                             constant: tensor(INT32, [], FLOAT16)
                         }
                     }
                 },
                 StructTypeProto {
                     name: "PARETOQ_UINT2"
                     structure: Structure {
                         field: {
                             name: "blocks"
                             type: array(
                                 struct(type_index=1),
                                 dimension=uint2_block_count
                             )
                         }
                         field: {
                             name: "tile_shape"
                             constant: tensor(INT64, [1], [128])
                         }
                         field: {
                             name: "data_type"
                             constant: tensor(INT32, [], FLOAT16)
                         }
                     }
                 }
             ]
         }

Tequila (ICLR 2026, 1.58 bits/weight, deadzone-free ternary)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Improved ternary QAT that removes the dead-zone around zero.
Storage is identical to TQ1_0/BitNet (ternary codebook).
The deadzone-free quantization function is only used during training.

.. tab-set::

   .. tab-item:: Specialized proto

      .. code-block:: text

         // Same storage as TQ1_0 / BitNet b1.58
         QuantizationProto {
             data_type: FLOAT16,
             tiling: TilingQuantizationProto {
                 tile_shape: [256],
                 elem_quant: [QuantizationProto {
                     codebook: CodebookProto {
                         scale_float: 1.0,
                         codebook_data: [-1.0, 0.0, 1.0],
                         packed_count: 5,
                         packed_bytes: 1
                     }
                 }]
             }
         }

         // Note: the difference with BitNet is in training only.
         // At inference, both use the same ternary codebook.

   .. tab-item:: Custom types

      .. code-block:: text

         ModelProto {
             struct_types: [
                 StructTypeProto {
                     name: "TEQUILA_BLOCK_256"
                     structure: Structure {
                         field: { name: "indices", type: array(UINT8, dimension=52) }
                         field: { name: "scale", type: array(FLOAT16, dimension=1) }
                         field: {
                             name: "codebook"
                             constant: tensor(FLOAT, [3], [-1.0, 0.0, 1.0])
                         }
                         field: {
                             name: "packed_count"
                             constant: tensor(INT32, [], 5)
                         }
                         field: {
                             name: "packed_bytes"
                             constant: tensor(INT32, [], 1)
                         }
                     }
                 },
                 StructTypeProto {
                     name: "TEQUILA"
                     structure: Structure {
                         field: {
                             name: "blocks"
                             type: array(
                                 struct(type_index=0),
                                 dimension=block_count
                             )
                         }
                         field: {
                             name: "tile_shape"
                             constant: tensor(INT64, [1], [256])
                         }
                         field: {
                             name: "data_type"
                             constant: tensor(INT32, [], FLOAT16)
                         }
                     }
                 }
             ]
         }

EETQ (INT8 weight-only per-channel)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Simple INT8 per-channel weight-only quantization (NetEase FuXi).
No calibration, no QAT. Uses optimized W8A16 GEMM kernels from
FasterTransformer / TensorRT-LLM.

.. tab-set::

   .. tab-item:: Specialized proto

      .. code-block:: text

         QuantizationProto {
             data_type: FLOAT16,
             linear: LinearUniformProto {
                 storage_type: INT8,
                 bits: 8,
                 symmetric: true,
                 scale_float: 0.0042,    // per-channel scale
                 zero_point: 0,
                 axis: 0                 // per output channel (row)
             }
         }

   .. tab-item:: Custom types

      .. code-block:: text

         StructTypeProto {
             name: "EETQ_INT8_PER_CHANNEL"
             structure: Structure {
                 field: { name: "values", type: array(INT8, dimension=N) }
                 field: {
                     name: "storage_type"
                     constant: tensor(INT32, [], INT8)
                 }
                 field: { name: "bits", constant: tensor(INT32, [], 8) }
                 field: { name: "symmetric", constant: tensor(BOOL, [], true) }
                 field: { name: "scale", constant: tensor(FLOAT, [], 0.0042) }
                 field: { name: "zero_point", constant: tensor(INT64, [], 0) }
                 field: { name: "axis", constant: tensor(INT32, [], 0) }
                 field: {
                     name: "data_type"
                     constant: tensor(INT32, [], FLOAT16)
                 }
             }
         }

EXL3 (improved EXL2, 2–6 bpw)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Successor to EXL2 with improved codebook and mixed-precision per layer.
Storage is structurally identical to EXL2 (variable bpw per layer using
nested ``TilingQuantizationProto``).

.. tab-set::

   .. tab-item:: Specialized proto

      .. code-block:: text

         // Same structure as EXL2, different calibration
         QuantizationProto {
             data_type: FLOAT16,
             tiling: TilingQuantizationProto {
                 tile_shape: [128],
                 elem_quant: [QuantizationProto {
                     linear: LinearUniformProto {
                         storage_type: UINT8,
                         bits: 3,               // varies per layer (2–6)
                         symmetric: false,
                         scale_float: 0.015,
                         zero_point: 4,
                         axis: -1
                     }
                 }]
             }
         }

   .. tab-item:: Custom types

      .. code-block:: text

         ModelProto {
             struct_types: [
                 StructTypeProto {
                     name: "EXL3_UINT3_BLOCK_128"
                     structure: Structure {
                         field: { name: "values", type: array(UINT8, dimension=48) }
                         field: {
                             name: "storage_type"
                             constant: tensor(INT32, [], UINT8)
                         }
                         field: { name: "bits", constant: tensor(INT32, [], 3) }
                         field: { name: "symmetric", constant: tensor(BOOL, [], false) }
                         field: { name: "scale", constant: tensor(FLOAT, [], 0.015) }
                         field: { name: "zero_point", constant: tensor(INT64, [], 4) }
                         field: { name: "axis", constant: tensor(INT32, [], -1) }
                     }
                 },
                 StructTypeProto {
                     name: "EXL3_LAYER"
                     structure: Structure {
                         field: {
                             name: "blocks"
                             type: array(
                                 struct(type_index=0),
                                 dimension=block_count
                             )
                         }
                         field: {
                             name: "tile_shape"
                             constant: tensor(INT64, [1], [128])
                         }
                         field: {
                             name: "data_type"
                             constant: tensor(INT32, [], FLOAT16)
                         }
                     }
                 }
             ]
         }

HQQ (Hybrid Quantization, 2–4 bits, mixed-precision per head)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Different bit-widths for Q, K, V projections in attention layers.
Each tensor gets its own ``QuantizationProto``; the example shows
a 2-bit weight with per-group scales.

.. tab-set::

   .. tab-item:: Specialized proto

      .. code-block:: text

         // Example: 2-bit weight with group size 64
         QuantizationProto {
             data_type: FLOAT16,
             tiling: TilingQuantizationProto {
                 tile_shape: [64],
                 elem_quant: [QuantizationProto {
                     linear: LinearUniformProto {
                         storage_type: UINT8,
                         bits: 2,
                         symmetric: false,
                         scale_float: 0.032,
                         zero_point: 2,
                         axis: -1
                     }
                 }]
             }
         }

         // In practice, Q proj may use 4-bit while V proj uses 2-bit.
         // Each tensor simply references a different QuantizationProto index.

   .. tab-item:: Custom types

      .. code-block:: text

         ModelProto {
             struct_types: [
                 StructTypeProto {
                     name: "HQQ_UINT2_BLOCK_64"
                     structure: Structure {
                         field: { name: "values", type: array(UINT8, dimension=16) }
                         field: {
                             name: "storage_type"
                             constant: tensor(INT32, [], UINT8)
                         }
                         field: { name: "bits", constant: tensor(INT32, [], 2) }
                         field: { name: "symmetric", constant: tensor(BOOL, [], false) }
                         field: { name: "scale", constant: tensor(FLOAT, [], 0.032) }
                         field: { name: "zero_point", constant: tensor(INT64, [], 2) }
                         field: { name: "axis", constant: tensor(INT32, [], -1) }
                     }
                 },
                 StructTypeProto {
                     name: "HQQ_HEAD"
                     structure: Structure {
                         field: {
                             name: "blocks"
                             type: array(
                                 struct(type_index=0),
                                 dimension=block_count
                             )
                         }
                         field: {
                             name: "tile_shape"
                             constant: tensor(INT64, [1], [64])
                         }
                         field: {
                             name: "data_type"
                             constant: tensor(INT32, [], FLOAT16)
                         }
                     }
                 }
             ]
         }

NVFP4 (NVIDIA FP4 E2M1 with FP8 scale, 4.5 bpw)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

NVIDIA's block floating-point FP4 format. Each element is E2M1
(2 exponent bits, 1 mantissa bit), with one FP8 E4M3 scale per
block of 16 elements.

.. tab-set::

   .. tab-item:: Specialized proto

      .. code-block:: text

         QuantizationProto {
             data_type: FLOAT16,
             tiling: TilingQuantizationProto {
                 tile_shape: [16],
                 elem_quant: [QuantizationProto {
                     floating_point: FloatingPointUniformProto {
                         sign_bits: 1,
                         exponent_bits: 2,
                         mantissa_bits: 1,
                         exponent_bias: 1,
                         has_inf: false,
                         has_nan: false,
                         packed_count: 2,      // 2 FP4 values per byte
                         packed_bytes: 1
                     }
                 }]
             }
         }

         // Dequantize: for each block of 16 elements
         // values = [fp_decode(v, 1, 2, 1, bias=1) for v in unpack_fp4(data)]
         // result = values * scale_fp8

   .. tab-item:: Custom types

      .. code-block:: text

         ModelProto {
             struct_types: [
                 StructTypeProto {
                     name: "NVFP4_E2M1_BLOCK_16"
                     structure: Structure {
                         field: {
                             name: "values"
                             type: array(FLOAT4E2M1, dimension=16)
                         }
                         field: {
                             name: "scale"
                             type: array(FLOAT8E4M3FN, dimension=1)
                         }
                         field: { name: "sign_bits", constant: tensor(INT32, [], 1) }
                         field: { name: "exponent_bits", constant: tensor(INT32, [], 2) }
                         field: { name: "mantissa_bits", constant: tensor(INT32, [], 1) }
                         field: { name: "exponent_bias", constant: tensor(INT32, [], 1) }
                         field: { name: "has_inf", constant: tensor(BOOL, [], false) }
                         field: { name: "has_nan", constant: tensor(BOOL, [], false) }
                         field: { name: "split_storage", constant: tensor(BOOL, [], false) }
                         field: { name: "packed_count", constant: tensor(INT32, [], 2) }
                         field: { name: "packed_bytes", constant: tensor(INT32, [], 1) }
                     }
                 },
                 StructTypeProto {
                     name: "NVFP4"
                     structure: Structure {
                         field: {
                             name: "blocks"
                             type: array(
                                 struct(type_index=0),
                                 dimension=block_count
                             )
                         }
                         field: {
                             name: "tile_shape"
                             constant: tensor(INT64, [1], [16])
                         }
                         field: {
                             name: "data_type"
                             constant: tensor(INT32, [], FLOAT16)
                         }
                     }
                 }
             ]
         }

QuaRot (rotational quantization, 4 bits)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Applies a Hadamard rotation before INT4 quantization to spread
outlier magnitudes evenly across channels. The rotation is stored
in ``RotationProto``.

.. tab-set::

   .. tab-item:: Specialized proto

      .. code-block:: text

         QuantizationProto {
             data_type: FLOAT16,
             pre_rotation: RotationProto {
                 type: HADAMARD,
                 dims: [4096]
             },
             tiling: TilingQuantizationProto {
                 tile_shape: [128],
                 elem_quant: [QuantizationProto {
                     linear: LinearUniformProto {
                         storage_type: UINT8,
                         bits: 4,
                         symmetric: true,
                         scale_float: 0.018,
                         zero_point: 0,
                         axis: -1
                     }
                 }]
             }
         }

         // Dequantize:
         // 1. values = unpack(data, 4 bits)
         // 2. result = values * scale
         // 3. result = hadamard_inverse(result)  ← RotationProto

   .. tab-item:: Custom types

      .. code-block:: text

         ModelProto {
             struct_types: [
                 StructTypeProto {
                     name: "QUAROT_UINT4_BLOCK_128"
                     structure: Structure {
                         field: { name: "values", type: array(UINT4, dimension=128) }
                         field: {
                             name: "storage_type"
                             constant: tensor(INT32, [], UINT8)
                         }
                         field: { name: "bits", constant: tensor(INT32, [], 4) }
                         field: { name: "symmetric", constant: tensor(BOOL, [], true) }
                         field: { name: "scale", constant: tensor(FLOAT, [], 0.018) }
                         field: { name: "zero_point", constant: tensor(INT64, [], 0) }
                         field: { name: "axis", constant: tensor(INT32, [], -1) }
                     }
                 },
                 StructTypeProto {
                     name: "QUAROT"
                     structure: Structure {
                         field: {
                             name: "blocks"
                             type: array(
                                 struct(type_index=0),
                                 dimension=block_count
                             )
                         }
                         field: {
                             name: "tile_shape"
                             constant: tensor(INT64, [1], [128])
                         }
                         field: {
                             name: "pre_rotation_type"
                             constant: tensor(INT32, [], 0)  // HADAMARD
                         }
                         field: {
                             name: "pre_rotation_dims"
                             constant: tensor(INT32, [1], [4096])
                         }
                         field: {
                             name: "data_type"
                             constant: tensor(INT32, [], FLOAT16)
                         }
                     }
                 }
             ]
         }

SmoothQuant (W8A8, smoothed INT8)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Shifts quantization difficulty from activations to weights via a
per-channel scaling transform. The smoothing factor is a diagonal
matrix stored as a rotation matrix (``PLAIN`` type).

.. tab-set::

   .. tab-item:: Specialized proto

      .. code-block:: text

         QuantizationProto {
             data_type: FLOAT,
             pre_rotation: RotationProto {
                 type: PLAIN,
                 dims: [4096],
                 matrix_index: 0          // index into ModelProto.rotation_matrices
             },
             linear: LinearUniformProto {
                 storage_type: INT8,
                 bits: 8,
                 symmetric: true,
                 scale_float: 0.0042,
                 zero_point: 0,
                 axis: 1                  // per-channel
             }
         }

         // Dequantize:
         // 1. result = int8_values * scale (per-channel)
         // 2. result = result @ diag(smooth_factors)^(-1)  ← RotationProto

   .. tab-item:: Custom types

      .. code-block:: text

         ModelProto {
             struct_types: [
                 StructTypeProto {
                     name: "SMOOTHQUANT_INT8"
                     structure: Structure {
                         field: {
                             name: "values"
                             type: array(INT8, dimension=N)
                         }
                         field: {
                             name: "storage_type"
                             constant: tensor(INT32, [], INT8)
                         }
                         field: { name: "bits", constant: tensor(INT32, [], 8) }
                         field: { name: "symmetric", constant: tensor(BOOL, [], true) }
                         field: { name: "scale", constant: tensor(FLOAT, [], 0.0042) }
                         field: { name: "zero_point", constant: tensor(INT64, [], 0) }
                         field: { name: "axis", constant: tensor(INT32, [], 1) }
                         field: {
                             name: "pre_rotation_type"
                             constant: tensor(INT32, [], 1)  // PLAIN
                         }
                         field: {
                             name: "pre_rotation_dims"
                             constant: tensor(INT32, [1], [4096])
                         }
                         field: {
                             name: "pre_rotation"
                             constant: tensor(FLOAT, [4096], smooth_factors)
                         }
                         field: {
                             name: "data_type"
                             constant: tensor(INT32, [], FLOAT)
                         }
                     }
                 }
             ]
         }

Pseudo-code
+++++++++++

The dequantize and quantize pseudo-code for each format is shown inline
in the corresponding proto section above. The top-level dispatcher is:

.. code-block:: python

    def dequantize(qtensor: QuantizedTensorProto, model: ModelProto) -> float[]:
        quant = model.quantizations[qtensor.quantized_type]
        data = qtensor.raw_data

        match quant.kind:
            case LinearUniformProto as q:     ...  # see LinearUniformProto section
            case CodebookProto as q:          ...  # see CodebookProto section
            case FloatingPointUniformProto:   ...
            case SparseQuantizationProto:     ...
            case LogUniformProto:             ...
            case FunctionUniformProto:        ...
            case TilingQuantizationProto:     ...
            case CastUniformProto:            ...
            case StructuredBlockUniformProto: ...

        # Apply post-rotation if present
        if quant.post_rotation:
            result = apply_rotation(result, quant.post_rotation, model)

        assert len(result) == product(qtensor.dims)
        return result
