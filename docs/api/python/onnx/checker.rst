onnx_light.onnx.checker
=======================

.. automodule:: onnx_light.onnx.checker
    :members:
    :imported-members:

Checker error codes
-------------------

``ValidationError.code`` identifies the failed rule. Compare codes rather than
messages: codes are stable across releases, while messages may change and contain
model-specific names.

.. code-block:: python

    import onnx_light.onnx.checker as checker

    try:
        checker.check_model(model)
    except checker.ValidationError as error:
        if error.code == "input-not-defined":
            print("An input is undefined or the graph is not topologically sorted.")

In C++, the identifier is returned by ``ValidationError::code()`` and constants
are defined in ``onnx_light::checker::error_code``. Rules without a code report
``None`` in Python and an empty string view in C++. Errors constructed in Python
also default to ``None``. Codes are added over time without changing existing
codes' names or meanings.

.. list-table::
    :header-rows: 1
    :widths: 35 65

    * - Code
      - Rule
    * - ``ir-version-missing``
      - The model does not set ``ir_version``.
    * - ``ir-version-unsupported``
      - The model's IR version is newer than the checker supports.
    * - ``metadata-props-duplicate-key``
      - Model metadata contains duplicate keys.
    * - ``required-field-missing``
      - A field required by the specification is absent.
    * - ``required-field-empty``
      - A field required to be non-empty is empty, such as a graph or initializer name.
    * - ``data-type-requires-newer-ir-version``
      - A tensor or type uses a data type introduced after the model's IR version.
    * - ``opset-import-missing``
      - A node's domain is not imported, or a model with IR version 3 or later imports no opset.
    * - ``operator-not-registered``
      - No schema exists at the imported version in a standard domain, or in any
        domain when ``check_custom_domain=True``.
    * - ``operator-deprecated``
      - The operator is deprecated at the imported opset version.
    * - ``node-schema-mismatch``
      - Input or output counts or required names, or duplicate, unknown, missing,
        or incorrectly typed attributes do not match the schema. Input and output
        types are not checked by this rule. Single-node shape inference reports
        the same code.
    * - ``input-not-defined``
      - A node input is not defined by an earlier node, graph input, initializer,
        or outer scope. It may be defined only later or never defined.
    * - ``not-ssa``
      - A value name is assigned more than once.
    * - ``graph-output-not-produced``
      - A graph output is not produced inside the graph.
    * - ``external-data-entry-invalid``
      - An external tensor also has inline data, lacks a location, or has an
        offset or length that is not a non-negative integer.
    * - ``external-data-location-invalid``
      - An external location is empty, absolute, unsafe, outside the model
        directory, a symbolic link, a hard link, or not a regular file.
    * - ``external-data-file-missing``
      - The external data file does not exist.
    * - ``external-data-out-of-range``
      - An offset or length exceeds the file, or the data is too small for the
        tensor's shape and type.

The local synthetic ``#`` external-data locations remain supported. Error codes
do not change validation order or the explicit external-location-check bypass.
