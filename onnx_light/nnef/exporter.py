"""Thin re-export shim over the C++ NNEF exporter (``_onnxpynnef``).

The original pure-Python implementation has been rewritten in C++; this
module preserves the public Python API and import paths.
"""

from __future__ import annotations

from onnx_light.onnx_py._onnxpy import nnef as _n

NNEFExportError = _n.NNEFExportError
NNEFGraph = _n.NNEFGraph
export_to_nnef = _n.export_to_nnef
to_nnef_text = _n.to_nnef_text
save_nnef = _n.save_nnef
register_op_converter = _n.register_op_converter
supported_ops = _n.supported_ops

# Dict-like view onto the C++ converter registry. Tests use this to clean
# up custom registrations via ``_CONVERTERS.pop(name, None)``.
_CONVERTERS = _n._ConverterRegistryView()

__all__ = [
    "NNEFExportError",
    "NNEFGraph",
    "export_to_nnef",
    "register_op_converter",
    "save_nnef",
    "supported_ops",
    "to_nnef_text",
]
