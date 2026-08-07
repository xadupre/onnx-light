
.. _l-next-steps-compiled-tensor:

CompiledTensorProto
===================

Stores prepacked tensors produced by a runtime (e.g. onnxruntime)
so that the prepacking step can be skipped on reload.
A ``CompiledTensorProto`` is a ``QuantizedTensorProto`` bound to a device.

.. code-block:: text

    message CompiledTensorProto {
        QuantizedTensorProto tensor = 1;   // quantized (or tiled) tensor data
        int32 device = 2;                  // index into ModelProto.devices
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

``QuantizedTensorProto.quantized_type`` is an index into ``quantizations``.
``CompiledTensorProto.device`` is an index into ``devices``.
``RotationProto.matrix_index`` is an index into ``rotation_matrices``.
