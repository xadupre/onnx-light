onnx_verify.h
=============

.. doxygenfile:: onnx_proto/onnx_verify.h
   :project: onnx-light

Persistent declarations
-----------------------

``VerifyPersistentBindings`` validates root-graph declarations using an optional
``StructTypeCatalogue``. ``VerifyGraph`` and ``VerifyModel`` call it automatically.
Bindings select whole inputs and outputs by exact graph names; dots and
backslashes are literal. Unknown names and duplicate input or output selections
are rejected. ``FeedbackState`` uses this same declaration validator.
String tensors cannot be persistent, including tensor fields and constants
inside whole structures or encoded layouts. ``ValidatePersistentType`` and
``ValidatePersistentStructType`` reuse the catalogue's recursive validation walk,
including reference resolution, cycle detection, depth limits and memoization.
``CompatiblePersistentTypes`` and ``CompatiblePersistentStructTypes`` expose
the same direct declaration compatibility checks to native runtime callers.
Compatibility alone does not make a value persistable: ordinary execution
also uses these checks, while persistent bindings additionally reject strings.
Unknown ranks and symbolic dimensions remain valid; conflicting known ranks,
concrete dimensions, dtypes and structured layout identities are rejected.
``StructTypeCatalogue::Build`` also accepts borrowed declarations directly, whose
lifetime and immutability requirements match the model overload.
