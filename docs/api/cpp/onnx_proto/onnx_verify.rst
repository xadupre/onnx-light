onnx_verify.h
=============

.. doxygenfile:: onnx_proto/onnx_verify.h
   :project: onnx-light

Persistent declarations
-----------------------

``VerifyPersistentBindings`` validates root-graph declarations using an optional
``StructTypeCatalogue``. ``VerifyGraph`` and ``VerifyModel`` call it automatically.
``ResolvePersistentBindingType`` returns a borrowed type selected by a literal
field-name array; it neither serializes types nor copies state buffers.
``CompatiblePersistentTypes`` and ``CompatiblePersistentStructTypes`` expose
the same direct declaration compatibility checks to native runtime callers.
Unknown ranks and symbolic dimensions remain valid; conflicting known ranks,
concrete dimensions, dtypes and structured layout identities are rejected.
``StructTypeCatalogue::Build`` also accepts borrowed declarations directly, whose
lifetime and immutability requirements match the model overload.
