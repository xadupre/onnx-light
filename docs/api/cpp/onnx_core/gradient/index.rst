gradient
========

The ``gradient`` sub-namespace of ``onnx_core`` (``core::gradient``) holds the
reverse-mode automatic differentiation engine shared by the ``onnx_gradient``
extension: the public entry points (:cpp:func:`GradientOfNodes`,
:cpp:func:`GradientOfFunction`), the per-operator gradient dispatcher, and the
common helpers used by the individual gradient builders.

.. toctree::
    :maxdepth: 1

    gradient
    grad_common
    grad_dispatcher
