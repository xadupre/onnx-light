parser.h
========

ONNX text-format parser declarations, including
:cpp:class:`onnx::ParserBase` (cursor-based tokenizer) and
:cpp:class:`onnx::OnnxParser` (builds protobuf structures from text).
Also defines utility maps :cpp:class:`onnx::PrimitiveTypeNameMap`,
:cpp:class:`onnx::AttributeTypeNameMap`, and :cpp:class:`onnx::KeyWordMap`.

.. doxygenfile:: parser.h
   :project: onnx-light
