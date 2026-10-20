dispatch_table.h
================

Shape functions opt into type-only tensor inputs with
``accepts_type_only_tensor_inputs`` when registering. This permits builder
inference to run without a ranked tensor descriptor; it must preserve unknown
rank rather than treating it as a scalar. ``Identity`` and ``Abs`` support this
mode. Functions that require ranked descriptors retain their existing readiness
checks.

.. doxygenfile:: onnx_core/shapes/dispatch_table.h
   :project: onnx-light
