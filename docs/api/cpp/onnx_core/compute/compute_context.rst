compute_context.h
=================

Serialized rewrites preserve existing tags for retained values throughout tag
seeding and inference, including results replaced by initializers. New value
names still participate in inference; removed names lose their tags. This
protection is scoped to rewriting and does not change ordinary tag propagation.

.. doxygenfile:: onnx_core/compute/compute_context.h
   :project: onnx-light
