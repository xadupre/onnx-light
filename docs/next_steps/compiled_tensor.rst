
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
        int32 quantized_type = 5;      // index into ModelProto.quantizations
        int32 device = 6;              // index into ModelProto.devices
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
        repeated QuantizationProto quantizations = <N>;
        repeated DeviceProto devices = <N+1>;
        repeated TensorProto rotation_matrices = <N+2>;
        ...
    }

``CompiledTensorProto.quantized_type``
and ``CompiledTensorProto.device`` are indices into these lists.
``RotationProto.matrix_index`` is an index into ``rotation_matrices``.
