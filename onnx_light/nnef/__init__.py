"""Export :mod:`onnx_light` models to the Khronos NNEF format.

NNEF (`Neural Network Exchange Format <https://www.khronos.org/nnef>`_)
is a textual description of a neural network graph together with a set
of binary tensor files (``*.dat``) for the parameters.  This sub-package
implements an ONNX → NNEF exporter that produces a directory layout
compatible with the NNEF v1.0 specification:

.. code-block:: text

    out_dir/
        graph.nnef
        weights.dat
        bias.dat
        ...

The exporter operates on any ``ModelProto``-like object: it duck-types
on the public attributes shared by :mod:`onnx` and :mod:`onnx_light`
messages (``graph.node``, ``graph.initializer``, ``graph.input``,
``graph.output`` and the usual ``op_type`` / ``input`` / ``output`` /
``attribute`` fields).  It therefore works equally well with models
loaded through :func:`onnx_light.onnx.load` or through the upstream
:mod:`onnx` package.

Only a subset of ONNX operators is supported out of the box (the most
common operators used by image classification networks).  Users can
register additional converters with :func:`register_op_converter`.

Example
-------
::

    from onnx_light.nnef import save_nnef

    save_nnef(model, "out_dir")
"""

from __future__ import annotations

from .exporter import (  # noqa: F401
    NNEFExportError,
    NNEFGraph,
    export_to_nnef,
    register_op_converter,
    save_nnef,
    supported_ops,
    to_nnef_text,
)
from .tensor_io import write_nnef_tensor  # noqa: F401
