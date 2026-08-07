
.. _l-next-steps-quantization:

Quantization
============

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
        }
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
        int32 data_type = 1;          // quantized element type (same enum as TensorProto.data_type)
        int32 bits = 2;               // number of bits (e.g. 4, 8)
        bool symmetric = 3;           // true if zero_point is always 0
        oneof scale {
            float scale_float = 4;    // scale as float
            int32 scale_int = 5;      // scale as shared exponent (value = q * 2^scale_int)
        }
        int64 zero_point = 6;         // quantization zero point
        int32 axis = 7;               // axis for per-channel, -1 if per-tensor
    }

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

FunctionUniformProto
^^^^^^^^^^^^^^^^^^^^

Custom quantization/dequantization defined by op names.
``data_type`` and ``bits`` describe the storage format of the quantized
data so the runtime can read the raw bytes correctly. The custom ops
referenced by ``quantize_op`` and ``dequantize_op`` handle the actual
conversion logic. The dequantize op takes a tensor of ``data_type``
and returns FLOAT32; the quantize op does the reverse.

.. code-block:: text

    message FunctionUniformProto {
        int32 data_type = 1;          // storage element type (how raw bytes are interpreted)
        int32 bits = 2;               // bits per element (for sub-byte packing)
        string quantize_op = 3;       // op: float32 -> data_type (e.g. "custom::Quantize")
        string dequantize_op = 4;     // op: data_type -> float32 (e.g. "custom::Dequantize")
    }

BlockQuantizationProto
^^^^^^^^^^^^^^^^^^^^^^

Recursive block quantization. Each block has a size and a nested
``QuantizationProto`` describing how elements within the block are
quantized. Nesting allows multi-level hierarchies (K-Quants, MXFP).

.. code-block:: text

    message BlockQuantizationProto {
        int32 block_size = 1;                          // elements per block at this level
        repeated QuantizationProto elem_quant = 2;     // one per block
    }

TilingQuantizationProto
^^^^^^^^^^^^^^^^^^^^^^^

Multi-dimensional block quantization. Extends ``BlockQuantizationProto``
to N dimensions: weights are partitioned into tiles of shape
``tile_shape`` along the given ``axes``, and each tile is quantized
independently. Useful for prepack layouts (MatMulNBits) where
quantization parameters vary along more than one axis.

.. code-block:: text

    message TilingQuantizationProto {
        repeated int64 tile_shape = 1;                 // block size per axis (e.g. [32, 128])
        repeated int32 axes = 2;                       // axes being tiled (e.g. [0, 1])
        QuantizationProto elem_quant = 3;              // quantization scheme (same for all tiles)
        repeated int32 perm = 4;                       // permutation of axes in memory layout
    }

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
            data_type: INT8, bits: 8, symmetric: true,
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
                    data_type: INT4, bits: 4,
                    symmetric: false, scale_float: 0.015, zero_point: 8, axis: -1
                }},
                QuantizationProto { linear: LinearUniformProto {
                    data_type: INT4, bits: 4,
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
                    data_type: INT4, bits: 4,
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
                                data_type: UINT2, bits: 2,
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
                                data_type: UINT4, bits: 3,
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
                                data_type: UINT4, bits: 4,
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
                                data_type: UINT8, bits: 5,
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
                                data_type: UINT8, bits: 6,
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
            data_type: INT8, bits: 8, symmetric: true,
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
            data_type: FLOAT8E4M3FN, bits: 8, symmetric: true,
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
                            data_type: INT4, bits: 3,
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
        pre_rotation: RotationProto { hadamard_size: 4096 },
        post_rotation: RotationProto { hadamard_size: 4096 }
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
                    data_type: INT4, bits: 2,
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
                    data_type: INT4, bits: 4,
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
            data_type: UINT4, bits: 4,
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
                data_type: UINT4, bits: 4,
                symmetric: true, scale_float: 0.0, zero_point: 0, axis: -1
            }},
            perm: [1, 0]   // N-major ordering in memory
        }
    }

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
    }

Pseudo-code
+++++++++++

Dequantization
^^^^^^^^^^^^^^

.. code-block:: python

    def dequantize(qtensor: QuantizedTensorProto, model: ModelProto) -> float[]:
        quant = model.quantizations[qtensor.quantized_type]
        data = qtensor.raw_data
        n_elements = product(qtensor.dims)

        match quant.kind:

            case LinearUniformProto as q:
                values = unpack(data, q.bits)
                if q.scale_float:
                    result = (values - q.zero_point) * q.scale_float
                else:
                    result = (values - q.zero_point) * (2 ** q.scale_int)

            case CodebookUniformProto as q:
                indices = unpack_base(data, len(q.codebook), q.packed_count, q.packed_bytes)
                result = [q.codebook[i] * q.scale for i in indices]

            case VectorCodebookUniformProto as q:
                result = zeros(n_elements)
                for k in range(q.num_codebooks):
                    indices = unpack(data[k], q.index_bits)
                    codebook_k = q.codebook_data[k * q.codebook_size * q.vector_size:]
                    for i, idx in enumerate(indices):
                        result[i*q.vector_size:(i+1)*q.vector_size] += codebook_k[idx]

            case FloatingPointUniformProto as q:
                bits_per_elem = q.sign_bits + q.exponent_bits + q.mantissa_bits
                raw_values = unpack(data, bits_per_elem, q.split_storage)
                result = [fp_decode(v, q.sign_bits, q.exponent_bits,
                                    q.mantissa_bits, q.exponent_bias)
                          for v in raw_values]

            case SparseQuantizationProto as q:
                result = dequantize_with(data.dense_part, q.base_quant)
                for (index, value) in data.outliers:
                    result[index] = value

            case LogUniformProto as q:
                raw = unpack(data, q.bits)
                if q.has_sign:
                    sign = extract_sign(raw)
                    magnitude = raw & ((1 << (q.bits - 1)) - 1)
                else:
                    sign = 1
                    magnitude = raw
                result = sign * (q.base ** (magnitude + q.offset))

            case FunctionUniformProto as q:
                tensor = unpack(data, q.bits, q.data_type)
                result = call_op(q.dequantize_op, tensor)

            case BlockQuantizationProto as q:
                result = []
                blocks = split(data, num_blocks=len(q.elem_quant))
                for i, block_data in enumerate(blocks):
                    block_values = dequantize_with(block_data, q.elem_quant[i])
                    result.extend(block_values)

        # Apply post-rotation if present
        if quant.post_rotation:
            result = apply_rotation(result, quant.post_rotation)

        assert len(result) == n_elements
        return result

Quantization
^^^^^^^^^^^^

.. code-block:: python

    def quantize(tensor: float[], quant: QuantizationProto) -> QuantizedTensorProto:
        # Apply pre-rotation if present
        if quant.pre_rotation:
            tensor = apply_rotation(tensor, quant.pre_rotation)

        match quant.kind:

            case LinearUniformProto as q:
                if q.scale_float:
                    values = round(tensor / q.scale_float) + q.zero_point
                else:
                    values = round(tensor / (2 ** q.scale_int)) + q.zero_point
                values = clip(values, 0, (1 << q.bits) - 1)
                raw_data = pack(values, q.bits)

            case CodebookUniformProto as q:
                scaled = tensor / q.scale
                indices = [nearest(scaled[i], q.codebook) for i in range(len(tensor))]
                raw_data = pack_base(indices, len(q.codebook), q.packed_count, q.packed_bytes)

            case BlockQuantizationProto as q:
                blocks = split(tensor, q.block_size)
                raw_data = b""
                for i, block in enumerate(blocks):
                    raw_data += quantize(block, q.elem_quant[i]).raw_data

            # ... other variants follow same pattern

        return QuantizedTensorProto(
            dims=shape(tensor), raw_data=raw_data,
            n_bytes=len(raw_data), quantized_type=<index>)

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
   * - FP6 LLM (TC-FPn)
     - 6.125
     - ``Block`` + ``FloatingPoint``
     - ✅
   * - FP8 E4M3
     - 8.0
     - ``Linear``
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
   * - QuIP#
     - 2.0
     - ``VectorCodebook`` + rotation
     - ✅
   * - SpQR
     - 3.4
     - ``Sparse`` + ``Block`` + ``Linear``
     - ✅
