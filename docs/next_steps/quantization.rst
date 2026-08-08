
.. _l-next-steps-quantization:

Quantization
============

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
     - ``VectorCodebook``
     - ✅
   * - Binary/XNOR (1-bit)
     - 1.0
     - ``Codebook``
     - ✅
   * - EXL2 (variable bpw)
     - 3.5
     - ``Block`` + ``Linear`` (multiple)
     - ✅
   * - EETQ (INT8 weight-only)
     - 8.0
     - ``Linear``
     - ✅
   * - EXL3 (improved EXL2)
     - 2–6
     - ``Block`` + ``Linear`` or ``Codebook``
     - ✅
   * - FP6 LLM (TC-FPn)
     - 6.125
     - ``Block`` + ``FloatingPoint``
     - ✅
   * - FP8 E4M3
     - 8.0
     - ``Linear``
     - ✅
   * - HQQ (mixed-precision per head)
     - 2–4
     - ``Block`` + ``Linear``
     - ✅
   * - INT4 AWQ (per-group)
     - 4.5
     - ``Block`` + ``Linear``
     - ✅
   * - INT4 GPTQ (per-group)
     - 4.5
     - ``Block`` + ``Linear``
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
     - ``Block`` + ``VectorCodebook``
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
     - ``Block`` + ``FloatingPoint``
     - ✅
   * - MXFP6 E3M2
     - 6.125
     - ``Block`` + ``FloatingPoint``
     - ✅
   * - NF4 (QLoRA)
     - 4.5
     - ``Codebook``
     - ✅
   * - NVFP4 (E2M1 + FP8 scale)
     - 4.5
     - ``Block`` + ``FloatingPoint``
     - ✅
   * - Q2_K
     - 2.625
     - ``Block`` × 2 + ``Linear``
     - ✅
   * - Q3_K
     - 3.4
     - ``Block`` × 2 + ``Linear``
     - ✅
   * - Q4_K
     - 4.5
     - ``Block`` × 2 + ``Linear``
     - ✅
   * - Q5_K
     - 5.5
     - ``Block`` × 2 + ``Linear``
     - ✅
   * - Q6_K
     - 6.6
     - ``Block`` × 2 + ``Linear``
     - ✅
   * - QuaRot (rotational quantization)
     - 4.0
     - ``Linear`` + ``RotationProto``
     - ✅
   * - QuIP#
     - 2.0
     - ``VectorCodebook`` + rotation
     - ✅
   * - SpQR
     - 3.4
     - ``Sparse`` + ``Block`` + ``Linear``
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

.. code-block:: text

    message QuantizedTensorProto {
        repeated int64 dims = 1;       // logical shape of the tensor
        bytes raw_data = 2;            // quantized payload
        int64 n_bytes = 3;            // byte size of raw_data
        int32 quantized_type = 4;      // index into ModelProto.quantizations
        optional QuantizationProto quantization = 5;  // inline if unique to this tensor
    }

QuantizationProto
+++++++++++++++++

.. code-block:: text

    // Describes how a tensor is quantized. Eight variants cover the
    // common families: linear affine/symmetric, scalar codebook lookup,
    // vector codebook (additive VQ), micro-float (exponent+mantissa),
    // sparse (outlier separation), logarithmic, custom function, and
    // recursive block-wise quantization.
    // BlockQuantizationProto can nest QuantizationProto to express
    // multi-level hierarchies (e.g. K-Quants, MXFP).
    // Optional pre/post rotations support QuIP#, SmoothQuant, etc.
    message QuantizationProto {
        oneof kind {
            LinearUniformProto linear = 1;
            CodebookUniformProto codebook = 2;
            VectorCodebookUniformProto vector_codebook = 3;
            FloatingPointUniformProto floating_point = 4;
            SparseQuantizationProto sparse = 5;
            LogUniformProto log = 6;
            FunctionUniformProto function = 7;
            BlockQuantizationProto block = 8;
            TilingQuantizationProto tiling = 13;
            IdentityProto identity = 14;
            CastUniformProto cast = 16;
            PackedLinearUniformProto packed_linear = 17;
            StructuredBlockUniformProto structured_block = 18;
        }
        int32 data_type = 15;              // dequantized element type (same enum as TensorProto.data_type)
        string doc_string = 9;             // human-readable description
        repeated StringStringEntryProto metadata_props = 10;  // arbitrary key-value metadata
        optional RotationProto pre_rotation = 11;   // rotation applied before quantization
        optional RotationProto post_rotation = 12;  // rotation applied after dequantization
    }

LinearUniformProto
^^^^^^^^^^^^^^^^^^

Classic affine/symmetric: ``value = (q - zero_point) * scale``.

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

CodebookUniformProto
^^^^^^^^^^^^^^^^^^^^

Lookup-table based: ``value = codebook[index] * scale``.

.. code-block:: text

    message CodebookUniformProto {
        oneof scale {
            float scale_float = 1;    // scale as float
            int32 scale_int = 2;      // scale as shared exponent
        }
        repeated float codebook = 3;  // lookup table values, len(codebook) = base
        int32 packed_count = 4;       // number of values packed
        int32 packed_bytes = 5;       // into this many bytes
    }

.. code-block:: python

    # Dequantize
    indices = unpack_base(data, len(q.codebook), q.packed_count, q.packed_bytes)
    result = [q.codebook[i] * q.scale for i in indices]

    # Quantize
    scaled = tensor / q.scale
    indices = [nearest(scaled[i], q.codebook) for i in range(len(tensor))]
    raw_data = pack_base(indices, len(q.codebook), q.packed_count, q.packed_bytes)

VectorCodebookUniformProto
^^^^^^^^^^^^^^^^^^^^^^^^^^

Additive vector codebook quantization (AQLM, residual VQ).
Each block of ``vector_size`` values is reconstructed as the sum
of lookups from ``num_codebooks`` codebooks.
``values[0:vector_size] = sum(codebooks[k][index_k] for k in range(num_codebooks))``

.. code-block:: text

    message VectorCodebookUniformProto {
        int32 num_codebooks = 1;      // number of additive codebooks (e.g. 2)
        int32 codebook_size = 2;      // entries per codebook (e.g. 256 for 8-bit index)
        int32 vector_size = 3;        // floats per codebook entry (e.g. 8)
        int32 index_bits = 4;         // bits per index (e.g. 8)
        repeated float codebook_data = 5;  // all codebooks concatenated:
                                           // num_codebooks * codebook_size * vector_size floats
    }

.. code-block:: python

    # Dequantize
    result = zeros(n_elements)
    for k in range(q.num_codebooks):
        indices = unpack(data[k], q.index_bits)
        codebook_k = q.codebook_data[k * q.codebook_size * q.vector_size:]
        for i, idx in enumerate(indices):
            result[i*q.vector_size:(i+1)*q.vector_size] += codebook_k[idx]

FloatingPointUniformProto
^^^^^^^^^^^^^^^^^^^^^^^^^

Micro-float quantization: ``value = (-1)^sign * 2^(exp - bias) * (1 + mantissa)``.
Covers FP6, FP4, MXFP and similar reduced-precision floating-point formats.

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

.. code-block:: text

    message SparseQuantizationProto {
        QuantizationProto base_quant = 1;  // quantization for non-outlier values
        int32 outlier_data_type = 2;       // data type for outlier values (e.g. FLOAT16)
        float outlier_threshold = 3;       // absolute value threshold for outlier detection
        float outlier_ratio = 4;           // fraction of values stored as outliers (e.g. 0.01)
    }

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

.. code-block:: text

    message LogUniformProto {
        int32 bits = 1;               // number of bits (e.g. 4, 8)
        float base = 2;              // logarithm base (e.g. 2.0)
        float offset = 3;            // exponent offset
        bool has_sign = 4;           // true if sign bit is stored separately
    }

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

.. code-block:: text

    message FunctionUniformProto {
        int32 storage_type = 1;       // storage element type (how raw bytes are interpreted)
        int32 bits = 2;               // bits per element (for sub-byte packing)
        string quantize_op = 3;       // op: data_type -> storage_type (e.g. "custom::Quantize")
        string dequantize_op = 4;     // op: storage_type -> data_type (e.g. "custom::Dequantize")
        optional BlockLayoutProto block_layout = 5;  // physical block structure (for introspection)
    }

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

BlockQuantizationProto
^^^^^^^^^^^^^^^^^^^^^^

Recursive block quantization. Each block has a size and a nested
``QuantizationProto`` describing how elements within the block are
quantized. Nesting allows multi-level hierarchies (K-Quants, MXFP).
If there are more blocks than entries in ``elem_quant``, the list
is cycled from the beginning (``elem_quant[i % len(elem_quant)]``).

.. code-block:: text

    message BlockQuantizationProto {
        int32 block_size = 1;                          // elements per block at this level
        repeated QuantizationProto elem_quant = 2;     // one per block
    }

.. code-block:: python

    # Dequantize
    blocks = split(data, q.block_size)
    result = []
    for i, block in enumerate(blocks):
        eq = q.elem_quant[i % len(q.elem_quant)]
        result.append(dequantize(block, eq))
    result = concat(result)

TilingQuantizationProto
^^^^^^^^^^^^^^^^^^^^^^^

Multi-dimensional block quantization. Extends ``BlockQuantizationProto``
to N dimensions: weights are partitioned into tiles of shape
``tile_shape`` along the given ``axes``, and each tile is quantized
independently. Useful for prepack layouts (MatMulNBits) where
quantization parameters vary along more than one axis.

.. code-block:: text

    message TilingQuantizationProto {
        repeated int64 tile_shape = 1;                 // block size per axis (0 = full tensor dim)
        repeated int32 axes = 2;                       // axes being tiled (e.g. [0, 1])
        QuantizationProto elem_quant = 3;              // quantization scheme (same for all tiles)
        repeated int32 perm = 4;                       // permutation of axes in memory layout
    }

.. code-block:: python

    # Dequantize
    shape = resolve_tile_shape(q.tile_shape, tensor_shape)
    tiles = split_tiles(data, shape, q.axes, q.perm)
    result = empty(tensor_shape)
    for coords, tile_data in tiles:
        result[coords] = dequantize(tile_data, q.elem_quant)
    if q.perm:
        result = inverse_permute(result, q.perm)

IdentityProto
^^^^^^^^^^^^^

No quantization. Data is stored as-is in its original data type.
Useful with ``TilingQuantizationProto`` to express a pure retiling
(block layout) without any value transformation.
The element type is given by ``QuantizationProto.data_type``.

.. code-block:: text

    message IdentityProto {}

.. code-block:: python

    # Dequantize
    result = interpret(data, quant_proto.data_type)

CastUniformProto
^^^^^^^^^^^^^^^^

Type conversion without quantization. Stores values cast from the
original type to a different type (e.g. float32 → bfloat16).
The source type is ``QuantizationProto.data_type``, the target type
is ``storage_type``.

.. code-block:: text

    message CastUniformProto {
        int32 storage_type = 1;        // target element type (same enum as TensorProto.data_type)
    }

.. code-block:: python

    # Dequantize
    result = cast(data, from_type=q.storage_type, to_type=quant_proto.data_type)

    # Quantize
    raw_data = cast(tensor, from_type=quant_proto.data_type, to_type=q.storage_type)

PackedLinearUniformProto
^^^^^^^^^^^^^^^^^^^^^^^^

Block-wise linear quantization where scales, zero points and corrections
are stored inside ``raw_data`` alongside the quantized weights.
Used for prepacked formats (CompInt8, QGEMM) where the runtime bakes
metadata into a single contiguous buffer.

.. code-block:: text

    message PackedLinearUniformProto {
        int32 storage_type = 1;        // quantized weight element type (e.g. UINT4, INT8)
        int32 bits = 2;                // bits per weight element
        bool symmetric = 3;           // true if zero_point is always 0
        int32 block_size = 4;          // quantization group size
        int32 axis = 5;                // quantization axis (-1 = per-tensor)
        int32 scale_type = 6;          // element type of embedded scales (e.g. FLOAT)
        int32 zero_point_type = 7;     // element type of embedded zero points (0 = absent)
        int32 correction_type = 8;     // element type of corrections (0 = absent)
        PackedLayout layout = 9;       // how regions are organized in the blob
    }

    enum PackedLayout {
        SEQUENTIAL = 0;               // [weights][scales][zero_points][corrections]
        INTERLEAVED = 1;              // per-block: [block_weights, scale, zp, correction] repeated
    }

.. code-block:: python

    # Dequantize
    if q.layout == SEQUENTIAL:
        weights, scales, zps, corrections = split_sequential(data, shape, q)
    else:
        weights, scales, zps, corrections = split_interleaved(data, shape, q)
    weights = unpack(weights, q.bits)
    for block_i in range(n_blocks):
        s = scales[block_i]
        zp = zps[block_i] if q.zero_point_type else 0
        c = corrections[block_i] if q.correction_type else 0
        result[block_i] = (weights[block_i] - zp) * s + c

StructuredBlockUniformProto
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Generic block quantization with explicit physical layout. Each block
contains named fields at known bit offsets, an optional codebook, an
index formula combining fields into codebook indices, and an optional
scatter pattern for non-contiguous element placement.
Covers exotic formats (STQ1_0, future formats) while keeping
dequantization fully deductible from the proto structure.

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
        int32 bit_offset = 2;          // offset in bits from block start
        int32 bit_width = 3;           // bits per element
        int32 count = 4;               // number of elements
        int32 data_type = 5;           // element type (0 = raw unsigned bits)
    }

    message BlockLayoutProto {
        int32 block_size = 1;                  // logical values per block
        int32 bytes_per_block = 2;             // physical bytes per block
        repeated BlockFieldProto fields = 3;   // physical fields in the block
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
        repeated float codebook_data = 2;              // codebook entries (flattened)
        int32 codebook_vector_size = 3;                // values per codebook entry
        repeated FieldWeightProto index_formula = 4;   // index = sum(field * multiplier)
        optional ScatterProto scatter = 5;             // output element placement (absent = contiguous)
    }

.. code-block:: python

    # Dequantize
    blocks = split_blocks(data, q.block_layout.bytes_per_block)
    result = []
    for block in blocks:
        fields = parse_fields(block, q.block_layout.fields)
        scale = fields[SCALE][0] if SCALE in fields else 1.0
        zp = fields[ZERO_POINT][0] if ZERO_POINT in fields else 0.0
        values = []
        for i in range(q.block_layout.block_size):
            idx = sum(fields[fw.field][i] * fw.multiplier for fw in q.index_formula)
            vec = q.codebook_data[idx * q.codebook_vector_size:][:q.codebook_vector_size]
            values.extend(vec)
        if q.scatter:
            values = inverse_scatter(values, q.scatter)
        result.append((values - zp) * scale)
    result = concat(result)

RotationProto
^^^^^^^^^^^^^

Pre/post rotation applied to the tensor (QuIP#, SmoothQuant).
Dequantization with rotation: ``values = post_rotation @ dequant(data) @ pre_rotation``.

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

Known quantization schemes
+++++++++++++++++++++++++++

INT8 symmetric (per-tensor)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

    QuantizationProto {
        linear: LinearUniformProto {
            storage_type: INT8, bits: 8, symmetric: true,
            scale_float: 0.02, zero_point: 0, axis: -1
        }
    }

INT4 asymmetric (GPTQ, per-group of 128)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a tensor of 1024 elements (8 groups):

.. code-block:: text

    QuantizationProto {
        block: BlockQuantizationProto {
            block_size: 128
            elem_quant: [
                QuantizationProto { linear: LinearUniformProto {
                    storage_type: INT4, bits: 4,
                    symmetric: false, scale_float: 0.015, zero_point: 8, axis: -1
                }},
                QuantizationProto { linear: LinearUniformProto {
                    storage_type: INT4, bits: 4,
                    symmetric: false, scale_float: 0.012, zero_point: 7, axis: -1
                }},
                // ... one per block (8 total)
            ]
        }
    }

NF4 (QLoRA / bitsandbytes)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

    QuantizationProto {
        codebook: CodebookUniformProto {
            scale_float: 1.0,
            codebook: [-1.0, -0.6962, -0.5251, -0.3949, -0.2844,
                       -0.1848, -0.0911, 0.0, 0.0796, 0.1609,
                       0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0],
            packed_count: 2, packed_bytes: 1
        }
    }

AWQ INT4 (per-group of 128)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

    QuantizationProto {
        block: BlockQuantizationProto {
            block_size: 128
            elem_quant: [
                QuantizationProto { linear: LinearUniformProto {
                    storage_type: INT4, bits: 4,
                    symmetric: false, scale_float: 0.01, zero_point: 8, axis: -1
                }},
                // ... one per block
            ]
        }
    }

Q2_K (llama.cpp, nested 256/16)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

    QuantizationProto {
        block: BlockQuantizationProto {
            block_size: 256
            elem_quant: [
                QuantizationProto {
                    block: BlockQuantizationProto {
                        block_size: 16
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

Q3_K (llama.cpp, nested 256/32)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

    QuantizationProto {
        block: BlockQuantizationProto {
            block_size: 256
            elem_quant: [
                QuantizationProto {
                    block: BlockQuantizationProto {
                        block_size: 32
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

Q4_K (llama.cpp, nested 256/32)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

    QuantizationProto {
        block: BlockQuantizationProto {
            block_size: 256
            elem_quant: [
                QuantizationProto {
                    block: BlockQuantizationProto {
                        block_size: 32
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

Q5_K (llama.cpp, nested 256/32)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

    QuantizationProto {
        block: BlockQuantizationProto {
            block_size: 256
            elem_quant: [
                QuantizationProto {
                    block: BlockQuantizationProto {
                        block_size: 32
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

Q6_K (llama.cpp, nested 256/16)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

    QuantizationProto {
        block: BlockQuantizationProto {
            block_size: 256
            elem_quant: [
                QuantizationProto {
                    block: BlockQuantizationProto {
                        block_size: 16
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

MXFP4 (OCP Microscaling, shared exponent per group of 32)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

    QuantizationProto {
        block: BlockQuantizationProto {
            block_size: 32
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

MXFP6 E3M2 (OCP Microscaling, 6-bit float per group of 32)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

    QuantizationProto {
        block: BlockQuantizationProto {
            block_size: 32
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

FP6 LLM (DeepSpeed TC-FPn, per-group of 128)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Same FP6 E3M2 quantization as MXFP6 but with split storage
(2-bit sign+exp and 4-bit mantissa stored separately) for
Tensor Core alignment.

.. code-block:: text

    QuantizationProto {
        block: BlockQuantizationProto {
            block_size: 128
            elem_quant: [
                QuantizationProto { floating_point: FloatingPointUniformProto {
                    sign_bits: 1, exponent_bits: 3, mantissa_bits: 2,
                    exponent_bias: 3, has_inf: false, has_nan: false,
                    split_storage: true, packed_count: 4, packed_bytes: 3
                }},
                // ... one per block
            ]
        }
    }

INT8 per-channel (classic CNN)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

    QuantizationProto {
        linear: LinearUniformProto {
            storage_type: INT8, bits: 8, symmetric: true,
            scale_float: 0.03, zero_point: 0, axis: 0
        }
    }

1.58-bit ternary (BitNet b1.58)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

    QuantizationProto {
        codebook: CodebookUniformProto {
            scale_float: 0.5,
            codebook: [-1.0, 0.0, 1.0],
            packed_count: 5, packed_bytes: 1
        }
    }

FP8 E4M3 (per-tensor)
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

    QuantizationProto {
        linear: LinearUniformProto {
            storage_type: FLOAT8E4M3FN, bits: 8, symmetric: true,
            scale_float: 1.0, zero_point: 0, axis: -1
        }
    }

AQLM 2×8 (Additive Quantization, 2 bits/weight)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

    QuantizationProto {
        vector_codebook: VectorCodebookUniformProto {
            num_codebooks: 2,
            codebook_size: 256,
            vector_size: 8,
            index_bits: 8,
            codebook_data: [...]  // 2 * 256 * 8 = 4096 floats
        }
    }

IQ1_S (llama.cpp, 1.56 bits/weight, E8 lattice)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Uses a 2048-entry vector codebook (E8 lattice grid, 8 values per entry).
Each sub-block of 8 weights is looked up with an 11-bit index.
Wrapped in a block of 256 for the shared FP16 scale.

.. code-block:: text

    QuantizationProto {
        block: BlockQuantizationProto {
            block_size: 256
            elem_quant: [
                QuantizationProto { vector_codebook: VectorCodebookUniformProto {
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

SpQR (Sparse Quantization, ~3.4 bits/weight)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Outliers (>1% of values) stored in FP16, rest in INT3 per-group.

.. code-block:: text

    QuantizationProto {
        sparse: SparseQuantizationProto {
            base_quant: QuantizationProto {
                block: BlockQuantizationProto {
                    block_size: 16
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

QuIP# (Vector quantization with Hadamard rotation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

    QuantizationProto {
        vector_codebook: VectorCodebookUniformProto {
            num_codebooks: 1,
            codebook_size: 256,
            vector_size: 8,
            index_bits: 8,
            codebook_data: [...]  // learned codebook, 256 * 8 = 2048 floats
        },
        pre_rotation: RotationProto { matrix_type: HADAMARD, dims: [4096, 4096] },
        post_rotation: RotationProto { matrix_type: HADAMARD, dims: [4096, 4096] }
    }

EXL2 (variable bits per layer, ~3.5 bpw average)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

EXL2 assigns different bit-widths per layer to hit a target average bpw.
Each layer gets its own ``QuantizationProto`` (via ``quantized_type`` index),
so a model may use several quantization entries — e.g. 2-bit for less
important layers and 4-bit for critical ones.

.. code-block:: text

    // Layer A (less important): 2-bit per group of 128
    QuantizationProto {
        block: BlockQuantizationProto {
            block_size: 128
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
        block: BlockQuantizationProto {
            block_size: 128
            elem_quant: [
                QuantizationProto { linear: LinearUniformProto {
                    storage_type: INT4, bits: 4,
                    symmetric: false, scale_float: 0.01, zero_point: 8, axis: -1
                }},
                // ... one per block
            ]
        }
    }

Log quantization (4-bit)
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

    QuantizationProto {
        log: LogUniformProto {
            bits: 4,
            base: 2.0,
            offset: -7.0,
            has_sign: true
        }
    }

Custom (plugin-based)
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

    QuantizationProto {
        function: FunctionUniformProto {
            storage_type: UINT4, bits: 4,
            quantize_op: "vendor::QuantizeV2",
            dequantize_op: "vendor::DequantizeV2"
        }
    }

MatMulNBits INT4 (onnxruntime, per-group of 32, tiled)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Weights of shape ``[K, N]`` are tiled into blocks of 32 along axis K
and 128 along axis N (SIMD width). All tiles share the same INT4
quantization scheme. ``perm: [1, 0]`` indicates N-major tile ordering
in memory for cache locality.

.. code-block:: text

    // Weight shape: [4096, 4096], group_size=32, N_tile=128
    QuantizationProto {
        tiling: TilingQuantizationProto {
            tile_shape: [32, 128],
            axes: [0, 1],
            elem_quant: QuantizationProto { linear: LinearUniformProto {
                storage_type: UINT4, bits: 4,
                symmetric: true, scale_float: 0.0, zero_point: 0, axis: -1
            }},
            perm: [1, 0]   // N-major ordering in memory
        }
    }

Block-tiled floats (no quantization)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Reorganizes a row-major float matrix into tiles of ``[32, 128]``
without any value transformation. Useful for prepack layouts
that expect block-tiled storage.

.. code-block:: text

    // Weight shape: [4096, 4096], tiled into [32, 128] blocks
    QuantizationProto {
        tiling: TilingQuantizationProto {
            tile_shape: [32, 128],
            axes: [0, 1],
            elem_quant: QuantizationProto { identity: IdentityProto {} }
        }
    }

STQ1_0 (Sherry, 1.25 bits/weight, ternary codebook)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

256 values per 42-byte block. Each 5-bit combined index (4-bit code +
1-bit sign) selects from a 32-entry codebook of 4 ternary values.
Output positions use a stride-16 scatter within each 64-value chunk.

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

.. code-block:: text

    // Weight shape: [M, N], stored column-major
    QuantizationProto {
        tiling: TilingQuantizationProto {
            tile_shape: [0, 0],
            axes: [0, 1],
            elem_quant: QuantizationProto { identity: IdentityProto {} },
            perm: [1, 0]
        }
    }

TQ1_0 (llama.cpp, 1.6 bits/weight, ternary)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pure ternary: each weight is {-1, 0, +1}. Five ternary values are packed
into one byte (3⁵ = 243 ≤ 255). One FP16 scale per block of 256 weights.

.. code-block:: text

    // Codebook: [-1, 0, +1], packed 5 per byte
    QuantizationProto {
        data_type: FLOAT16,
        block: BlockQuantizationProto {
            block_size: 256,
            elem_quant: [QuantizationProto {
                codebook: CodebookUniformProto {
                    scale_float: 1.0,       // scale is per-block, stored separately
                    codebook: [-1.0, 0.0, 1.0],
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

TQ2_0 (llama.cpp, 2 bits/weight, ternary)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ternary with 2 bits per weight (simpler packing, 1 unused state).
Mapping: 0 → -1, 1 → 0, 2 → +1.

.. code-block:: text

    // Codebook: [-1, 0, +1], 2 bits per value
    QuantizationProto {
        data_type: FLOAT16,
        block: BlockQuantizationProto {
            block_size: 256,
            elem_quant: [QuantizationProto {
                codebook: CodebookUniformProto {
                    scale_float: 1.0,
                    codebook: [-1.0, 0.0, 1.0],
                    packed_count: 4,        // 4 values per byte (2 bits each)
                    packed_bytes: 1
                }
            }]
        }
    }

    // Dequantize: same as TQ1_0 but simpler unpack (2 bits, not base-3)

BitNet b1.58 (Microsoft, 1.58 bits/weight, native ternary)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Model trained natively with ternary weights {-1, 0, +1}.
Storage is identical to TQ1_0 (5 ternary values per byte).
The difference is in training (QAT), not in the storage format.

.. code-block:: text

    // Same storage as TQ1_0
    QuantizationProto {
        data_type: FLOAT16,
        block: BlockQuantizationProto {
            block_size: 256,
            elem_quant: [QuantizationProto {
                codebook: CodebookUniformProto {
                    scale_float: 1.0,
                    codebook: [-1.0, 0.0, 1.0],
                    packed_count: 5,
                    packed_bytes: 1
                }
            }]
        }
    }

ParetoQ (Meta, 1.58–2 bits/weight, SEQ ternary/2-bit)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Quantization-aware training with Stretched Elastic Quantization (SEQ).
The storage format is standard ternary or 2-bit; the custom SEQ function
is used only during training. At inference, the model uses a standard
codebook. If the SEQ rounding function is needed at inference, use
``FunctionUniformProto``.

.. code-block:: text

    // Ternary variant (same storage as TQ1_0)
    QuantizationProto {
        data_type: FLOAT16,
        block: BlockQuantizationProto {
            block_size: 256,
            elem_quant: [QuantizationProto {
                codebook: CodebookUniformProto {
                    scale_float: 1.0,
                    codebook: [-1.0, 0.0, 1.0],
                    packed_count: 5,
                    packed_bytes: 1
                }
            }]
        }
    }

    // 2-bit variant (4 levels)
    QuantizationProto {
        data_type: FLOAT16,
        block: BlockQuantizationProto {
            block_size: 128,
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

Tequila (ICLR 2026, 1.58 bits/weight, deadzone-free ternary)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Improved ternary QAT that removes the dead-zone around zero.
Storage is identical to TQ1_0/BitNet (ternary codebook).
The deadzone-free quantization function is only used during training.

.. code-block:: text

    // Same storage as TQ1_0 / BitNet b1.58
    QuantizationProto {
        data_type: FLOAT16,
        block: BlockQuantizationProto {
            block_size: 256,
            elem_quant: [QuantizationProto {
                codebook: CodebookUniformProto {
                    scale_float: 1.0,
                    codebook: [-1.0, 0.0, 1.0],
                    packed_count: 5,
                    packed_bytes: 1
                }
            }]
        }
    }

    // Note: the difference with BitNet is in training only.
    // At inference, both use the same ternary codebook.

EETQ (INT8 weight-only per-channel)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Simple INT8 per-channel weight-only quantization (NetEase FuXi).
No calibration, no QAT. Uses optimized W8A16 GEMM kernels from
FasterTransformer / TensorRT-LLM.

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

EXL3 (improved EXL2, 2–6 bpw)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Successor to EXL2 with improved codebook and mixed-precision per layer.
Storage is structurally identical to EXL2 (variable bpw per layer using
nested ``BlockQuantizationProto``).

.. code-block:: text

    // Same structure as EXL2, different calibration
    QuantizationProto {
        data_type: FLOAT16,
        block: BlockQuantizationProto {
            block_size: 128,
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

HQQ (Hybrid Quantization, 2–4 bits, mixed-precision per head)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Different bit-widths for Q, K, V projections in attention layers.
Each tensor gets its own ``QuantizationProto``; the example shows
a 2-bit weight with per-group scales.

.. code-block:: text

    // Example: 2-bit weight with group size 64
    QuantizationProto {
        data_type: FLOAT16,
        block: BlockQuantizationProto {
            block_size: 64,
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

NVFP4 (NVIDIA FP4 E2M1 with FP8 scale, 4.5 bpw)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

NVIDIA's block floating-point FP4 format. Each element is E2M1
(2 exponent bits, 1 mantissa bit), with one FP8 E4M3 scale per
block of 16 elements.

.. code-block:: text

    QuantizationProto {
        data_type: FLOAT16,
        block: BlockQuantizationProto {
            block_size: 16,
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

QuaRot (rotational quantization, 4 bits)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Applies a Hadamard rotation before INT4 quantization to spread
outlier magnitudes evenly across channels. The rotation is stored
in ``RotationProto``.

.. code-block:: text

    QuantizationProto {
        data_type: FLOAT16,
        pre_rotation: RotationProto {
            type: HADAMARD,
            dims: [4096]
        },
        block: BlockQuantizationProto {
            block_size: 128,
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

SmoothQuant (W8A8, smoothed INT8)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Shifts quantization difficulty from activations to weights via a
per-channel scaling transform. The smoothing factor is a diagonal
matrix stored as a rotation matrix (``PLAIN`` type).

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
            case CodebookUniformProto as q:   ...  # see CodebookUniformProto section
            case VectorCodebookUniformProto:  ...
            case FloatingPointUniformProto:   ...
            case SparseQuantizationProto:     ...
            case LogUniformProto:             ...
            case FunctionUniformProto:        ...
            case BlockQuantizationProto:      ...
            case TilingQuantizationProto:     ...
            case IdentityProto:              ...
            case CastUniformProto:           ...
            case PackedLinearUniformProto:    ...
            case StructuredBlockUniformProto: ...

        # Apply post-rotation if present
        if quant.post_rotation:
            result = apply_rotation(result, quant.post_rotation, model)

        assert len(result) == product(qtensor.dims)
        return result
