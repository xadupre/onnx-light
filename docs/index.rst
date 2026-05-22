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

onnx without protobuf
+++++++++++++++++++++

- **Files larger than 2 GB** – The standard ``onnx`` package relies on
  protobuf, which enforces a 2 GB message-size limit and cannot load or save
  models that exceed that threshold. ``onnx-light`` bypasses protobuf entirely
  and supports arbitrarily large ONNX files.
- **External-data / multi-file models** – Large models can be split across a
  ``.onnx`` structure file and one or more separate weight files.  Pass a
  ``location`` argument to ``onnxl.save`` to route tensor weights to a
  separate file, and use ``load_external_data=True`` when loading them back.
  The ``max_external_file_size`` option automatically partitions weights across
  multiple files (``model.data``, ``model.data.1``, …) when a single file
  would exceed the given byte limit.
- **Parallel loading** – Tensor weights can be read in parallel using multiple
  threads, which significantly reduces wall-clock load time for large models.
- **Zero-copy parsing** – When parsing from an in-memory bytes buffer, the
  ``no_copy=True`` option makes each tensor's ``raw_data`` point directly into
  the source bytes without allocating an extra copy.  This eliminates one
  ``malloc + memcpy`` per tensor initializer:

  .. code-block:: python

      import onnx_light.onnx as onnxl

      serialized = open("model.onnx", "rb").read()   # keep alive!
      model = onnxl.load(serialized, no_copy=True)
      # tensor.raw_data now points into 'serialized' – no extra copy

  .. warning::
      The original bytes object **must** remain alive for as long as the
      returned model is in use.  This constraint does not apply to the
      standard ``onnx`` package.

- **Encrypted save / load** – Models can be encrypted with AES-256-CBC
  (PBKDF2-HMAC-SHA256 key derivation) and saved to a single self-contained
  ``.onnxc`` file, or serialized to an in-memory ``bytes`` object.  This
  feature is unique to ``onnx-light`` and requires that the package was built
  with OpenSSL support:

  .. code-block:: python

      import onnx_light.onnx as onnxl

      # File-based
      onnxl.save_encrypted(model, "model.onnxc", key="my_passphrase")
      model = onnxl.load_encrypted("model.onnxc", key="my_passphrase")

      # In-memory bytes (no file I/O)
      blob = onnxl.save_encrypted_string(model, key="my_passphrase")
      model = onnxl.load_encrypted_string(blob, key="my_passphrase")

Getting started
+++++++++++++++

Install the package in editable mode:

.. code-block:: bash

    pip install -e .[dev] -v

or

.. code-block:: bash

    python setup.py build_ext --inplace

To speed up compilation with multiple threads, pass ``--parallel`` (or ``-j``)
with the number of jobs:

.. code-block:: bash

    python setup.py build_ext --inplace --parallel 8

By default, ``python setup.py build_ext`` auto-enables parallel builds
(``--parallel <cpu_count>``) unless ``CMAKE_BUILD_PARALLEL_LEVEL`` is already set.

Alternatively, when installing with pip, control parallel builds using the
``CMAKE_BUILD_PARALLEL_LEVEL`` environment variable:

.. code-block:: bash

    CMAKE_BUILD_PARALLEL_LEVEL=8 pip install -e .[dev]

Run a quick check:

.. code-block:: bash

    python -c "import onnx_light; print(onnx_light.__version__)"

Build and run the C++ unit tests from the editable build:

With ``pip install``:

.. code-block:: bash

    pip install -C build-dir=build -C cmake.build-type=Debug -C cmake.define.ONNX_LIGHT_BUILD_TESTS=ON -e .[dev]
    ctest --test-dir build --output-on-failure

With ``setup.py``:

.. code-block:: bash

    python setup.py build_ext --inplace --build-temp build --cpp-tests
    ctest --test-dir build --output-on-failure

On multi-config generators such as Visual Studio, add the matching
configuration to ``ctest``: use ``-C Debug`` when the build was configured with
``cmake.build-type=Debug``, and ``-C Release`` after ``python setup.py
build_ext --cpp-tests``.

Load a model with parallel tensor parsing:

.. code-block:: python

    import onnx_light.onnx

    model = onnx_light.onnx.load("model.onnx", num_threads=4)
    print(model.ir_version)

Source code: `https://github.com/xadupre/onnx-light <https://github.com/xadupre/onnx-light>`_

.. toctree::
    :maxdepth: 1
    :caption: Contents

    design/index
    howto/index
    api/index
    operators/index
    examples
    misc
