.. _l-howto-graph-builder-basics:

:html_theme.sidebar_secondary.remove:

Build and optimize a graph with GraphBuilder
=============================================

:class:`onnx_light.onnx_core.graph_builder.GraphBuilder` incrementally builds
ONNX graphs while resolving operator schemas, inferring shapes, and assigning
unique value names. This walkthrough uses its compact authoring API and then
optimizes a second graph with the standard pattern library.

Create and export a model
-------------------------

Select every opset before adding its nodes. The empty domain is the standard
ONNX domain; a custom operator requires an explicit non-empty domain and an
imported version. It does not require a schema registered on the local machine.

The example includes:

* ``Add`` as a standard operator;
* ``Clip`` with an omitted optional ``min`` input;
* variadic ``Sum`` inputs;
* both outputs of ``TopK``;
* a schema-less ``com.example::CustomNormalize`` operator.

.. runpython::

    from pathlib import Path
    import tempfile

    import numpy as np

    import onnx_light.onnx as onnxl
    import onnx_light.onnx.checker as checker
    from onnx_light.onnx import TensorProto
    from onnx_light.onnx_core.graph_builder import GraphBuilder

    g = GraphBuilder("authoring")
    g.set_opset_version("", 18)
    g.set_opset_version("com.example", 7)

    x = g.inp("X", TensorProto.FLOAT, [4])
    bias = g.init(np.array([1, 2, 3, 4], dtype=np.float32), name="bias")
    added = g.op.Add(x, bias)
    clipped = g.op.Clip(added, None, np.array(6, dtype=np.float32))
    summed = g.op.Sum(clipped, x, bias)
    k = g.init(np.array([2], dtype=np.int64), name="k")
    values, indices = g.op.TopK(
        summed, k, outputs=["values", "indices"], axis=0
    )
    normalized = g.op.CustomNormalize(
        values,
        domain="com.example",
        outputs="Y",
        epsilon=1e-5,
    )
    g.out(normalized, TensorProto.FLOAT, [2])

    model = g.to_onnx("model")
    checker.check_model(model)

    with tempfile.TemporaryDirectory() as temporary_directory:
        path = Path(temporary_directory) / "model.onnx"
        onnxl.save(model, path)
        loaded = onnxl.load(path)
        checker.check_model(loaded)
        assert loaded.SerializeToString() == model.SerializeToString()

    opsets = {opset.domain or "ai.onnx": opset.version for opset in model.opset_import}
    custom = next(node for node in model.graph.node if node.domain == "com.example")
    print(f"opsets: {opsets}")
    print(f"custom node: {custom.domain}::{custom.op_type}")
    print(f"TopK outputs: {values}, {indices}")
    print("model validated and round-tripped")

``g.inp`` declares and returns an input name, ``g.init`` adds a NumPy
initializer, ``g.op.<Operator>`` adds a node, and ``g.out`` declares an output.
Operator inputs can be value names, NumPy arrays, or ``None`` for an omitted
optional input. The ``outputs`` option accepts one name, a sequence of names, or
a positive output count.

The compact helpers delegate to the explicit
:meth:`~onnx_light.onnx_core.graph_builder.GraphBuilder.make_input`,
:meth:`~onnx_light.onnx_core.graph_builder.GraphBuilder.make_initializer`,
:meth:`~onnx_light.onnx_core.graph_builder.GraphBuilder.make_node`, and
:meth:`~onnx_light.onnx_core.graph_builder.GraphBuilder.make_output` methods.
Those ``make_*`` methods are the complete low-level contract for generated code
and advanced authoring.

Borrow a NumPy initializer without copying
-----------------------------------------

``builder.init(array, copy=False)`` borrows the array's payload instead of
allocating or copying a weight buffer. The default ``copy=True`` keeps the
existing independent-copy behavior::

    import numpy
    from onnx_light.onnx_core.graph_builder import GraphBuilder

    weights = numpy.ones((1024, 1024), dtype=numpy.float32)
    builder = GraphBuilder("borrowed")
    name = builder.init(weights, name="weights", copy=False)
    builder.out(name)
    model = builder.to_onnx()
    del weights, builder
    serialized = model.SerializeToString()

The array is retained by a reference-counted payload owner, not just by the
builder. Exported graphs/models, builders constructed from those models, and
``TensorProto.CopyFrom`` / ``copy.copy(tensor)`` retain that owner. The array
reference is released with the Python GIL held when the last borrowed payload
is destroyed or replaced. Serialization produces independent bytes;
``ModelProto.CopyFrom`` retains its existing serialization-based deep-copy
behavior.

Only C-contiguous, little-endian arrays are accepted. Supported dtypes are
``bool``, signed/unsigned 8/16/32/64-bit integers, ``float16/32/64``, and
``complex64/128``. Non-contiguous/Fortran-only layouts, incompatible byte order,
object/string, structured, and other dtypes raise an exception; there is no
fallback to a copy. Scalars, empty arrays, contiguous views, and read-only
arrays are supported.

**Mutation contract:** writable arrays are allowed and remain writable.
Mutations through the array or its aliases remain visible in borrowed tensors
and models. Finish mutations before optimization or creating an execution
session, which may cache derived values. Do not resize/reallocate storage while
borrowed. Marking an array read-only
does not freeze existing writable aliases. Use the default ``copy=True`` when
an immutable snapshot independent of the source is required.
Shape analysis retains the tensor's type and dimensions but does not decode
managed borrowed payloads to cache numeric bounds or shape values.

Transfer an existing initializer
-------------------------------

``builder.make_initializer(tensor)`` copies a ``TensorProto`` and leaves it
unchanged. When the caller no longer needs the tensor, the explicit
``builder.make_initializer_move(tensor)`` method transfers it instead::

    import numpy
    from onnx_light.onnx import TensorProto, numpy_helper
    from onnx_light.onnx_core.graph_builder import GraphBuilder

    builder = GraphBuilder("transfer")
    tensor = numpy_helper.from_array(numpy.ones(1024, dtype=numpy.float32), name="weight")
    name = builder.make_initializer_move(tensor)
    assert name == "weight"
    assert tensor.SerializeToString() == TensorProto().SerializeToString()

On success the source is cleared to an empty, reusable ``TensorProto``. All
Python references to that same source object observe the cleared state.
Validation failures leave the source unchanged. Both methods use the same name
and public-input default checks, shape inference, and value annotations.
External-data metadata is transferred without reading the external file.

The C++ equivalent is ``builder.MakeInitializerMove(std::move(tensor))``;
``MakeInitializer(const TensorProto&)`` retains its copying behavior. Moving
preserves the backing allocation of owned ``raw_data`` and retains the owner
token of borrowed ``raw_data`` without materializing it. This avoids the
insertion copy; subsequent graph export operations retain their existing
copying behavior.

Structured values and encoded initializers
-----------------------------------------

Native models can declare reusable ``StructTypeProto`` types and store
``EncodedValueProto`` initializers. These are onnx-light protobuf extensions,
not ordinary ONNX tensors. A declaration describes one record; each payload's
byte extent determines its record count. Different payload lengths can
therefore share one type.

.. code-block:: python

    from onnx_light.onnx import (
        EncodedValueProto,
        StructTypeProto,
        TensorProto,
        TypeProto,
        ValueInfoProto,
    )
    from onnx_light.onnx import helper
    from onnx_light.onnx_core.graph_builder import GraphBuilder

    # Each record contains two UINT8 elements.
    pair = StructTypeProto()
    pair.name = "BytePair"
    pair.type_id = 1
    pair.array.dimension = 2
    pair.array.element_type.CopyFrom(
        helper.make_tensor_type_proto(TensorProto.UINT8, [])
    )

    builder = GraphBuilder("encoded_records")
    builder.set_opset_version("", 18)
    builder.make_struct_type(pair)

    value = EncodedValueProto()
    value.name = "records"
    value.struct_type.type_ref = 1
    value.raw_data = b"\x01\x02\x03\x04"
    builder.make_encoded_initializer(value)

    output_type = TypeProto()
    output_type.struct_type.type_ref = 1
    output = ValueInfoProto()
    output.name = "records"
    output.type.CopyFrom(output_type)
    builder.make_output(output)

    native_model = builder.to_onnx("model")
    restored = GraphBuilder(native_model)
    layout = restored.shapes.get_encoded_layout("records")
    assert layout.payload_bytes == 4
    assert layout.record_count == 2

Declarations also support named typed fields, tensor-valued constant fields,
nested arrays and bit packing. Constant fields occupy no payload bits.
``make_struct_type`` registers a model-scoped nonzero ``type_id``; references
must resolve in that model, including its nested subgraphs. Declare referenced
types before their users when constructing a model incrementally. Importing a
model loads its complete catalogue before its graph.

``ShapesContext`` keeps the structured type, optional decoded
``logical_type``, and physical layout together. Its ``get_type``,
``resolve_struct_type`` and ``get_encoded_layout`` queries distinguish field
types from payload geometry. A logical tensor descriptor does not decode the
stored bytes or make an encoded initializer an ordinary tensor constant.
The affine branch retains its storage type, scales, zero point, axis and
block size; its logical tensor type must have concrete dimensions.
An encoded initializer sharing a graph input's name is an overridable
default: inference validates its layout but retains the public input type
and symbolic shape rather than treating the default bytes as a constant.
``Identity`` preserves the structured descriptor, including structured types
nested in sequence, optional and map containers. ``If`` branches and
``Loop``-carried values preserve matching structured types; encoded values
must also have identical layouts and payloads. Standard tensor operators
do not implicitly decode encoded inputs. Custom consumers need registered
shape inference; unsupported structured operations and control-flow merges
fail explicitly rather than guessing a layout.

For external payloads, set ``data_location`` to ``TensorProto.EXTERNAL``
and provide ``external_data`` entries for ``location`` and ``length``
(and optionally ``offset``). Validation checks the declared extent, not
the file's contents. Native C++ callers can also use
``EncodedValueProto::set_raw_data_with_deleter`` to retain shared buffer
ownership across builder and context copies.

``build_graph`` and ``to_onnx("model")`` preserve the native extensions.
Use a model, rather than a standalone graph, to retain shared declarations.
``to_standard_model`` explicitly rejects structured constructs: lowering
to standard tensors and operators is not implemented. ORT serialization
also rejects them. Do not pass native structured models to an upstream
consumer expecting standard ONNX support.

Optimize and replay a rewrite
-----------------------------

The following model contains a redundant ``Cast`` from ``float`` to ``float``.
Selecting only the standard ``Cast`` pattern makes the result deterministic:
one :class:`~onnx_light.onnx_core.optimization.LocalRewriting` replaces it with
``Identity``.

.. runpython::

    import onnx_light.onnx.checker as checker
    from onnx_light.onnx import TensorProto
    from onnx_light.onnx_core.graph_builder import GraphBuilder
    from onnx_light.onnx_core.optimization import (
        GraphGraph,
        replay,
        standard_patterns,
    )
    source = GraphBuilder("optimization")
    source.set_opset_version("", 18)
    x = source.inp("X", TensorProto.FLOAT, [4])
    y = source.op.Cast(x, outputs="Y", to=TensorProto.FLOAT)
    source.out(y, TensorProto.FLOAT, [4])
    original = source.to_onnx("model")

    builder = GraphBuilder(original)
    graph = GraphGraph(
        builder,
        standard_patterns(["Cast"]),
        use_global_patterns=False,
    )
    rewrites, report = graph.optimize(report=True)
    optimized_graph = builder.build_graph()

    assert len(rewrites) == 1
    assert report.rewrites == 1
    cast_report = next(item for item in report.patterns if item.pattern_name == "Cast")
    print(
        f"Cast: {cast_report.matches} match(es) over "
        f"{cast_report.attempts} attempt(s)"
    )
    print(repr(rewrites[0]))

    replayed_graph = replay(original, rewrites)
    assert replayed_graph.SerializeToString() == optimized_graph.SerializeToString()

    optimized_model = builder.to_onnx("model")
    checker.check_model(optimized_model)
    print("replay reproduced the optimized graph")

``report`` aggregates attempts, matches, rejections, and timings by pattern.
Each returned ``LocalRewriting`` is also a replayable record of the matched and
added nodes, their positions, initializer changes, and value renames.
