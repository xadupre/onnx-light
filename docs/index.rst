onnx-light
==========

.. image:: https://github.com/xadupre/onnx-light/actions/workflows/ci_core.yml/badge.svg
    :target: https://github.com/xadupre/onnx-light/actions/workflows/ci_core.yml
    :alt: core

.. image:: https://github.com/xadupre/onnx-light/actions/workflows/build.yml/badge.svg
    :target: https://github.com/xadupre/onnx-light/actions/workflows/build.yml
    :alt: build

.. image:: https://github.com/xadupre/onnx-light/actions/workflows/mypy.yml/badge.svg
    :target: https://github.com/xadupre/onnx-light/actions/workflows/mypy.yml
    :alt: mypy

.. image:: https://github.com/xadupre/onnx-light/actions/workflows/docs.yml/badge.svg
    :target: https://github.com/xadupre/onnx-light/actions/workflows/docs.yml
    :alt: Documentation

.. image:: https://github.com/xadupre/onnx-light/actions/workflows/style.yml/badge.svg
    :target: https://github.com/xadupre/onnx-light/actions/workflows/style.yml
    :alt: Style

.. image:: https://github.com/xadupre/onnx-light/actions/workflows/pyrefly.yml/badge.svg
    :target: https://github.com/xadupre/onnx-light/actions/workflows/pyrefly.yml
    :alt: pyrefly

.. image:: https://github.com/xadupre/onnx-light/actions/workflows/spelling.yml/badge.svg
    :target: https://github.com/xadupre/onnx-light/actions/workflows/spelling.yml
    :alt: Spelling

.. image:: https://codecov.io/gh/xadupre/onnx-light/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/xadupre/onnx-light

.. image:: https://img.shields.io/github/repo-size/xadupre/onnx-light
    :target: https://github.com/xadupre/onnx-light

onnx without protobuf.

Getting started
===============

Install the package in editable mode:

.. code-block:: bash

    pip install -e .[dev]

Run a quick check:

.. code-block:: bash

    python -c "import onnx_light; print(onnx_light.__version__)"

Load a model with parallel tensor parsing:

.. code-block:: python

    import onnx_light.onnx as onnxl

    model = onnxl.load("model.onnx", parallel=True, num_threads=4)
    print(model.ir_version)

Source code: `https://github.com/xadupre/onnx-light <https://github.com/xadupre/onnx-light>`_

.. toctree::
    :maxdepth: 1
    :caption: Contents

    api/index
    auto_examples/index
