
.. _l-next-steps-compiled-tensor:

CompiledTensorProto
===================

Stores prepacked tensors produced by a runtime (e.g. onnxruntime)
so that the prepacking step can be skipped on reload.

.. code-block:: text

    message CompiledTensorProto {
        repeated int64 dims = 1;       // dimensions of the tensor
        bytes raw_data = 2;            // prepacked blob as produced by the runtime
        int32 data_type = 3;           // element data type (same enum as TensorProto.data_type)
        int64 n_bytes = 4;             // byte size of raw_data
        int32 layout = 5;              // index into ModelProto.layouts
        int32 quantized_type = 6;      // index into ModelProto.quantizations
        int32 device = 7;              // index into ModelProto.devices
    }

LayoutProto
+++++++++++

.. code-block:: text

    message LayoutProto {
        string name = 1;       // layout identifier (e.g. "MlasPackedGemm")
        string runtime = 2;   // runtime that produces this layout (e.g. "onnxruntime")
        int64 version = 3;    // layout format version
    }

DeviceProto
+++++++++++

.. code-block:: text

    message DeviceProto {
        string type = 1;    // device type (e.g. "cpu", "cuda")
        int32 index = 2;    // device index (e.g. 0)
    }

ModelProto extension
++++++++++++++++++++

.. code-block:: text

    message ModelProto {
        ...
        repeated LayoutProto layouts = <N>;
        repeated QuantizationProto quantizations = <N+1>;
        repeated DeviceProto devices = <N+2>;
        repeated TensorProto rotation_matrices = <N+3>;
        ...
    }

``CompiledTensorProto.layout``, ``CompiledTensorProto.quantized_type``
and ``CompiledTensorProto.device`` are indices into these lists.
``RotationProto.matrix_index`` is an index into ``rotation_matrices``.
