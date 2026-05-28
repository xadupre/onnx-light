expressions.h
=============

Symbolic dimension-expression utilities for ONNX shape inference.

Provides a lightweight AST-based library for parsing, simplifying,
evaluating, and renaming symbolic shape expressions such as those produced
during ONNX shape inference (e.g. ``"2*batch//batch"`` → ``"batch"``).

See :doc:`/api/python/optim/expressions` for the Python interface.

.. doxygenfile:: expressions.h
   :project: onnx-light
