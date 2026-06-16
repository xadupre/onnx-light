parser.h
========

ONNX text-format parser declarations, including
:cpp:class:`onnx_light::ParserBase` (cursor-based tokenizer) and
:cpp:class:`onnx_light::OnnxParser` (builds protobuf structures from text).
Also defines utility maps :cpp:class:`onnx_light::PrimitiveTypeNameMap`,
:cpp:class:`onnx_light::AttributeTypeNameMap`, and :cpp:class:`onnx_light::KeyWordMap`.

.. doxygenfile:: parser.h
   :project: onnx-light
