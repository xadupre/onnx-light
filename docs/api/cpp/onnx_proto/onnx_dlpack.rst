onnx_dlpack.h
=============

``ReleaseDLPack(TensorProto&)`` returns a standard ``DLManagedTensor*`` with
independent storage and metadata. It requires only ``lib_onnx_proto`` and the
installed header, not Python, nanobind, or a DLPack runtime.

.. warning::

   This is a **destructive transfer**: it immediately removes ``raw_data``
   (null pointer, zero size, false presence). The source tensor and any
   initializer/model containing it cannot use that payload until reassigned.
   All other fields are preserved. The descriptor and storage survive source
   destruction; neither its deleter nor user-provided storage callbacks may
   access a destroyed source.

The caller hands the descriptor to one consumer or calls its ``deleter`` once.
Consumers must treat storage as read-only. Borrowed storage requires a shared
lifetime owner. Validation/allocation failure leaves the source intact.
Existing aliases retain their original lifetime restrictions; a prior
non-destructive view does not keep transferred owned storage alive.

.. code-block:: cpp

   #include "onnx_dlpack.h"

   DLManagedTensor *exported = onnx_light::ReleaseDLPack(tensor);
   // The source may now be destroyed or assigned a new payload.
   // Pass exported to a DLPack consumer, or release it exactly once:
   exported->deleter(exported);

.. doxygenfile:: onnx_dlpack.h
   :project: onnx-light

DLPack C ABI
------------

The vendored header is from DLPack v1.1
(``https://github.com/dmlc/dlpack/tree/v1.1``), with only repository
clang-format formatting changes. Its Apache-2.0 license is retained alongside
the header and installed under ``share/licenses/onnx_light/dlpack``.
The existing recursive public-header installation includes this header.
This version supplies the legacy ABI and all supported dtype codes without
the unrelated exchange-function API added in v1.2.

.. doxygenfile:: dlpack/dlpack.h
   :project: onnx-light
