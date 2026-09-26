.. _l-howto-persistent-feedback:

Persistent input/output feedback
===============================

Feedback state retains selected model outputs as inputs for the next call.
The graph declares the relationship in ``GraphProto.persistent_bindings``;
types come from its final input/output declarations. The caller supplies
initial values, not a separate feedback mapping. Execution uses the existing
session and allocator infrastructure, not a separate executor.

Each bound input must have **exactly one value-use** in the graph. Every node
input position counts, including read-only operations such as ``Shape`` and
two positions of the same node. A graph output that directly returns that input
also counts, as do captures in nested graph attributes. The check counts
references statically, including references in mutually exclusive branches.
Input declarations, bindings and ``value_info`` metadata do not count as uses.
Both ONNX validation and runtime graph-plan construction enforce this rule;
graphs without persistent bindings retain their ordinary sharing semantics.

The native :cpp:class:`onnx_light::core::runtime::PersistentValueState` uses the
existing runtime execution and value ownership contracts. Create one state
per independent request. Python initialization/reset, C++ ownership transfer,
state forwarding and state-value access retain buffer owners without copying payloads.
Shapes, metadata and owner handles may be copied. Kernels can allocate new
computed results; the state layer does not duplicate those results merely
to retain or return them. The CPU Attention append optimization described
below reduces kernel allocations and prefix copies independently of this
zero-copy state forwarding.

.. warning::

   Inputs, retained state and returned views can share the same payload.
   Callers and kernels must not modify that payload while shared or retained.
   ``state.values`` is a shared view, not an independently mutable snapshot.
   Metadata such as a returned tensor's name and shape is independent.

The model is immutable for the entire bound session lifetime. State creation
and execution do not serialize, clone or hash the model to check for changes.
To rewrite the graph, create a new state/session after rewriting instead.

A basic feedback loop
---------------------

The Python binding is available from the native runtime module:

.. code-block:: python

    import numpy
    from onnx_light.onnx_lib import parser
    from onnx_light.onnx_py._onnxpykernels import runtime

    model = parser.parse_model(
        '<ir_version: 10, opset_import: ["" : 18]>'
        "accumulate (float[2] delta, float[2] past) => (float[2] present)"
        "{ present = Add(delta, past) }"
    )
    binding = model.graph.persistent_bindings.add()
    binding.input_name = "past"
    binding.output_name = "present"
    context = runtime.RuntimeContext(
        runtime.KernelContext(runtime.default_opset(18))
    )
    initial = numpy.zeros(2, dtype=numpy.float32)
    state = runtime.PersistentValueState(model, initial={"past": initial})
    delta = numpy.ones(2, dtype=numpy.float32)
    first = state.run(context, {"delta": delta})
    second = state.run(context, {"delta": delta})
    numpy.testing.assert_array_equal(
        numpy.from_dlpack(second["present"]),
        [2, 2],
    )
    state.reset({"past": initial})
    state.close()

The mapping is **whole input destination to whole output source**. An input
is entirely persistent or entirely supplied by current feeds, never partly
both. Binding names and initial/current-feed/``state.values`` dictionary keys
are exact, literal graph names, with no path syntax or escaping.
``"request.cache"`` names the graph input literally named ``request.cache``;
it cannot select a field of ``request``. Unknown names are rejected.
Nested dictionaries represent complete structured values, not partial feeds.

Inputs accept runtime ``Tensor`` objects, contiguous CPU NumPy arrays and
compatible DLPack producers such as CPU PyTorch tensors. Noncontiguous,
byte-swapped or unsupported representations raise an error rather than
being copied. PyTorch tensors requiring gradients must be detached by the
caller before DLPack export; ``detach()`` shares storage.

Selected outputs and ``state.values`` contain tensors with retained storage owners.
They remain valid after callers drop their input references, after another
run, or after reset/close. Unsupported ownerless output storage is rejected
rather than silently copied for a selected output; an allocator cannot recycle a live retained
allocation.
The binding keeps the model alive. The supplied context configures allocators
and custom kernels; each call executes in a fresh child context, rather than
leaving old feeds or intermediate values in the caller's context.
The binding also retains supplied contexts so cached kernel allocator
references remain valid. Reuse the same context for a state's calls.
Initializers use model-backed views when their representation is directly
readable; other numeric representations and strings use normal conversion.
String tensors cannot be persistent, including nested tensor fields and string
constants in a selected whole structure or encoded layout. Catalogue references
are checked recursively. Declarations fail before state initialization, and
runtime retention rejects string payloads rather than copying them.
Ordinary nonpersistent string feeds and outputs remain supported and use normal
materialized string storage.

Persistence applies only to the declared whole outputs, not the entire context.
Nonpersistent outputs keep normal allocator and materialization behavior.
There is no alternate kernel dispatcher: for example, tensor ``Identity`` still
computes an ordinary output rather than promising to alias its input.

The corresponding C++ entry points are:

.. code-block:: cpp

    using namespace onnx_light::core::runtime;

    auto *binding = model.mutable_graph()->add_persistent_bindings();
    binding->set_input_name("past");
    binding->set_output_name("present");
    PersistentValueState state(
        model,
        {{"past", RuntimeValue(Tensor::FromFloat("past", {2}, {0.f, 0.f}))}});
    RuntimeContext context(KernelContext(18));
    auto outputs = state.Run(
        context,
        {{"delta", RuntimeValue(Tensor::FromFloat("delta", {2}, {1.f, 1.f}))}});
    state.Reset(
        {{"past", RuntimeValue(Tensor::FromFloat("past", {2}, {0.f, 0.f}))}});
    state.Close();

The C++ constructor and ``Reset`` accept initial maps by value. Build an owned
``RuntimeValueMap`` and pass ``std::move(initial)`` for zero-copy transfer.
Passing an lvalue uses ordinary C++ copy semantics. To share an existing owned
tensor explicitly, use ``std::move(tensor).RetainStorage()`` once, then
``BorrowView()`` on the returned owner-backed view. Const reads never move or
promote storage.

C++ callers register the operator kernels as usual before executing the model
(see :doc:`register_builtin_operators`). The model and configured allocators
must outlive the state/session using them. With the reference-based C++
constructor, the model must also outlive any returned views of its initializer
storage. Use the ``shared_ptr<const ModelProto>`` constructor to retain the
model automatically in such views; the Python binding retains its model
automatically.

Lifecycle and validation
------------------------

* Construct the state **after** graph rewrites. Every graph-declared input and output
  must exist in the final model, with compatible types and shape constraints.
  A removed or changed input/output needs an updated graph binding and a new state.
* Supply initial contents for every feedback destination. Each subsequent
  call supplies the remaining whole inputs; current feeds must not override
  retained inputs. Duplicate binding input names or output names are rejected.
* State advances only after successful execution and validation of the next
  values. A failed or cancelled call must not publish a partial update.
* Reset explicitly supplies new initial contents. Closing releases retained
  values. Calls, resets and closes on the same state must not overlap.
* Independent states have separate state containers. They can explicitly share
  read-only input storage; neither may mutate a shared buffer.

Publication replaces owner handles atomically after validation. Failure or
cancellation leaves the old state available; it does not require backup
copies. This guarantee does not roll back external writes that violate the
read-only contract. Persistent declarations currently belong to the root
graph; declarations inside control-flow subgraphs are rejected.

Cancellation uses the existing task-completion primitive:

.. code-block:: python

    completion = runtime.TaskCompletion()
    completion.cancel("request no longer needed")
    # state.run(context, feeds, completion) now rejects the cancelled call.

A completion is single-use. A pending completion can also be cancelled from
another thread while a call runs. Already executing kernels finish normally,
but cancellation winning the publication race prevents the state update.
Successful publication completes the token, so cancelling it afterwards is
an error. A failed call leaves the previous valid state available for retry.
Execution releases the Python GIL; use independent contexts for concurrent
requests, and do not mutate model/context configuration during a call.

Structured feedback
-------------------

A binding retains a whole input, including every dynamic field of a structured
value. To retain a cache while supplying fresh tokens, declare ``cache`` and
``tokens`` as separate graph inputs and bind ``cache`` to a whole ``next_cache``
output. The struct declaration comes from
``StructTypeProto`` in the model; persistence is a property of the feedback
declaration in ``GraphProto``, not a flag on the type or encoded payload.

This is equivalent to manually taking the selected outputs from each
stateless invocation and passing them to the next invocation. Separate unselected
outputs, such as logits, are not part of retained state. Fields inside a selected
output are all retained.

Python represents named structs as nested dictionaries. For a model declaring
structured ``cache`` and ``next_cache`` values and a separate ``tokens`` input:

.. code-block:: python

    binding = model.graph.persistent_bindings.add()
    binding.input_name = "cache"
    binding.output_name = "next_cache"
    state = runtime.PersistentValueState(model, {"cache": initial_cache})
    output = state.run(context, {"tokens": tokens})

Custom kernels use ``context.get_value(name)`` and
``context.put_value(name, value)`` to exchange structured values; ordinary
tensor kernels continue to use the existing tensor API. In C++, structured
and encoded edges live in ``RuntimeContext::values()`` as ``RuntimeValue``
objects containing existing tensors or ``EncodedValueProto`` payloads.
Inline structured encoded payloads can be retained as whole values; external
payloads must first be loaded. The native API supports tensors, named
structs, typed sequences and inline structured or affine encodings. Affine
values are checked against their declared logical tensor type without decoding.
Persistent values always have concrete dimensions. For an encoded tensor,
``logical_type`` describes its decoded shape, not an unresolved symbolic shape:
missing ranks or non-concrete dimensions are rejected. Symbolic dimensions remain
valid in the model's input/output declarations.
Map, optional, sparse and opaque state remain unsupported and are rejected
explicitly, including when nested in a sequence or structure.
For the supported types, the native correspondence is:

* ``tensor_type``: a runtime tensor, or an encoded value with a matching logical
  tensor type.
* ``struct_type``: named runtime fields, or an encoded value with a compatible
  storage type.
* ``sequence_type``: runtime elements recursively checked against ``elem_type``.

``RuntimeValue.elements`` is a ``RuntimeSequence`` with immutable element
metadata and structural sharing. Copying a sequence shares its tree; appending
or replacing an element copies only a logarithmic path. Read access uses
``elements[i]`` or ``elements.at(i)``. To edit an element, obtain its
``BorrowView()``, change that independent descriptor and call
``elements.Set(i, std::move(value))``. Existing snapshots remain unchanged.
Payload owners are shared and payloads remain read-only. Deep copies still
produce independent payloads.

Validation reuses results for unchanged sequence subtrees and merges their
concrete symbolic bindings, rather than walking historical elements on every
invocation. Memo tables hold weak references, so they do not prolong page or
allocator lifetimes. Catalogue-dependent encoded retention still rechecks the
supplied catalogue.

These are ``TypeProto`` contracts, not a claim that every ``SequenceProto``,
``MapProto`` or ``OptionalProto`` has a native persistent representation.
Python feedback supports tensors, named structs, inline encoded values and
dedicated ``PagedCacheProto`` values. Arbitrary sequence conversion remains
unsupported.
``If`` and model-local functions forward selected whole output names and move
those results without persistence-related copies. Function attributes,
``Loop`` and ``Scan`` use their ordinary runtime implementations: their normal
computation/transport costs remain, but unrelated operators are not prohibited.

``GraphBuilder`` preserves and validates persistence declarations during
import/export and supported rewrites. Direct edits to graph input/output
names require corresponding binding edits; dangling names are rejected.
An export to standard ONNX
that cannot preserve this contract must be rejected, not silently strip
the bindings. Explicit model saving serializes the declarations, not the
current request's retained state; execution itself does not serialize them.

Contiguous CPU Attention feedback
--------------------------------

Declare ``past_key <- present_key`` and ``past_value <- present_value`` using
the ordinary graph bindings. The native CPU ``Attention`` consumer can then
retain extra allocation capacity for subsequent appends. There is no separate
cache identifier, state mapping or executor.

Internally, ``PersistentValueState`` retains ``PersistentValue`` objects, with a
``PersistentTensor`` at each tensor leaf. ``PersistentTensor`` composes an ordinary
``Tensor`` with certified allocation capacity; it does not inherit from ``Tensor``.
The runtime receives ordinary tensor views and separate, move-only ``AppendLease``
objects for eligible root bindings. Neither tensor copies nor borrowed views carry
capacity metadata or write permissions. Child function and subgraph contexts do
not inherit these permissions.

The reservation API is operator-independent:

1. A kernel calls ``RuntimeContext::ReservePersistentAppend`` with the desired
   result shape, append axis and input/output slots. The context resolves the
   exact declared binding and selects the output allocator.
2. ``PersistentTensor::AppendLease::Reserve`` handles layout eligibility,
   geometric growth and prefix relocation. It either reuses spare capacity or
   allocates a new contiguous buffer and copies only the committed prefix.
   Unsupported layouts return no reservation so the kernel can use its
   ordinary implementation. Each lease accepts only one attempt, including
   attempts declined for disabled capacity or unsupported layouts. A second
   attempt raises an error instead of allocating another buffer.
3. The kernel initializes the entire ``AppendReservation::writable_bytes()``
   span directly. There is no temporary tail tensor required by this API.
4. ``RuntimeContext::CommitPersistentAppend`` checks the declared initialized
   byte count, seals the candidate and returns an ordinary tensor view. This
   does not publish the state: ``PersistentValueState`` still validates all outputs
   and publishes them together only after successful completion.

A producer can compute new elements directly into that span. Attention instead
copies the current K/V inputs, which already exist as model inputs, straight into
the reserved region. Those copies remain necessary; the reservation API does not
add an intermediate tensor or a second copy.

This storage implementation is contiguous. A future ``PersistentPagedTensor``
would own a different allocation policy and expose page-aware writable regions;
paged storage and paged Attention kernels are not implemented here.

The reusable layout is dense rank-four ``FLOAT`` with shape
``[1, 1, valid_length, head_size]`` for each K/V tensor. Query tensors can
have multiple heads (multi-query attention). The tensor's sequence dimension
and logical byte extent describe only valid tokens, never spare capacity.
K and V may have different head sizes.
The Attention node must directly consume and produce the bound root-graph
K/V inputs and outputs. Intermediate tensors, function/control-flow transport
and unmatched input/output pairs keep ordinary kernel concatenation; they do
not acquire append permissions merely because another graph output is retained.

``RuntimeSessionOptions::persistent_tensor_initial_capacity`` selects the initial
capacity along the kernel's append axis (32 by default; tokens for Attention);
zero disables reservations. ``PersistentTensor`` grows capacity geometrically
when necessary, using the selected allocator without an alternate allocator or
automatic retry after allocation failure. This option affects ``PersistentValueState``
execution, not ordinary stateless ``RuntimeSession`` calls.

Python exposes the same option as a keyword-only constructor argument and a
read/write property on ``runtime.RuntimeSessionOptions``. It accepts a
nonnegative integer representable as C++ ``size_t``. Pass the options to
``PersistentValueState`` when constructing the state:

.. code-block:: python

    options = runtime.RuntimeSessionOptions(persistent_tensor_initial_capacity=64)
    state = runtime.PersistentValueState(model, initial, options=options)

    # Disables reservations for a separate state, without changing the first state.
    options.persistent_tensor_initial_capacity = 0
    ordinary_state = runtime.PersistentValueState(model, initial, options=options)

Options are copied at construction; changing the bundle later does not change
an existing state.

Reuse is deliberately conservative:

* Only internally created append buffers are eligible. An arbitrary borrowed
  NumPy/DLPack buffer, even one with an owner token, does not grant write access.
  Importing returned tensors or ``Values()`` views into a new state, or through
  ``Reset``, retains their bytes without copying but does not import append
  capacity. Their first append allocates a new certified buffer.
* Ownership is checked before creating invocation-local aliases. Keeping a
  previous output or ``Values()`` view alive prevents reuse of that allocation.
  The next result instead receives a fresh allocation; the old view's bytes,
  shape and lifetime do not change.
* An eligible append writes only the new token range. It does not move or
  rewrite the valid prefix. The previous state's logical extent remains
  unchanged until successful publication. Capacity is published only when the
  returned tensor still matches the kernel's candidate owner, pointer, type,
  shape and logical byte extent.
* Capacity exhaustion makes ``PersistentTensor`` allocate a larger buffer and
  copy the valid prefix once. State publication still only transfers owner
  handles.
* Multiple batches or KV heads use ordinary dense concatenation: increasing
  the sequence dimension changes the stride between heads, so prefix-preserving
  tail append is not possible in that layout. Half-precision promotion and
  other unsupported reuse paths keep their ordinary computation semantics.
* Empty prefixes and empty appended chunks contribute no copied bytes.
  Zero-width value caches use the dense fallback and keep their declared
  shapes, including the empty final dimension of the Attention output.

Cancellation is a publication gate, not kernel preemption. A failed or
cancelled invocation can have written unused tail bytes, but cannot change the
previous state's valid prefix or length. A retry must fully write its new
token range. Reset replaces the retained owners; close releases them. Existing
returned views remain readable after either operation. Use independent states
and contexts for concurrent requests; sharing retained owners disables unsafe
reuse rather than making either request mutate the other's state.

Opt-in storage auditing through runtime events
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Storage auditing uses the existing runtime event API. Set
``RuntimeContextOptions::events_enabled = true`` and read ``context.events()``.
Every context has a shared event log. With events disabled (the default),
persistence works identically but creates no storage audit events or counters
and takes no event-recording locks.
Diagnostic metadata is not copied into function or half-precision
scratch contexts on this path.

``RuntimeEventAction::kPersistentStorage`` identifies storage reports. Each
``RuntimeEvent`` directly records the work in ``storage_allocations``,
``storage_allocated_bytes``, ``storage_prefix_copied_bytes``,
``storage_append_copied_bytes`` and ``storage_reuse_count``.
Kernels and contiguous reservations use
``RuntimeContext::RecordEvent`` with action ``kPersistentStorage`` to record
this work in the existing shared log. Producing new elements directly into the
writable region does not count as copying them.

These are explicit reports, not automatic counters for every runtime
allocation or tensor copy. Attention currently reports K/V construction on
both reusable and ordinary dense paths. Other consumers can use the same
event API without adding operator-specific state to ``RuntimeContext``.

State forwarding and kernel storage construction have different costs.
Pointer identity at retention, invocation and publication boundaries verifies
that the state layer does not copy tensor payloads. Feedback invocations,
subgraphs, functions and half-precision scratch contexts share the caller's
event log. Events are visible as soon as they are recorded, including work
preceding a failure or cancellation; no forwarding or end-of-scope merge occurs.
Runtime recording serializes appends from concurrent children. Read or modify
``events()`` only when no other context is recording or clearing the log.
``ClearEvents()`` clears the shared log for all these contexts. Independently
constructed contexts keep independent logs. Allocation failures while recording
propagate to the caller, just like other runtime allocation failures.
Attention's storage reports exclude its
score/output allocations, arithmetic workspace, feed construction, or
half-precision conversion.

Sum fields from the event list when totals are needed. Call
``context.ClearEvents()`` before a run to obtain per-token reports; otherwise
events accumulate in that context, including runs of different feedback states.
Resetting or closing a state does not clear the caller's log.

``event.storage_allocated_bytes`` counts requested storage capacity,
not physical heap allocations: an I/O arena may satisfy a request from its free
lists. The existing ``event.allocated_bytes`` and ``event.peak_bytes`` fields
still describe allocator live and peak memory and have not changed meaning.
The decode example enables events explicitly, so its timing includes auditing.

For one new token with ``FLOAT`` K/V head sizes ``Dk`` and ``Dv``, appending
copies ``4 * (Dk + Dv)`` bytes. Reuse within capacity allocates no new KV
buffers and copies zero prefix bytes. Growth or an outstanding external alias
requires a new buffer for each affected K/V tensor and copies its valid prefix.
For ``B`` batches, ``H`` KV heads and a prefix of ``L`` tokens, the dense
fallback allocates two result buffers and copies
``4 * B * H * L * (Dk + Dv)`` prefix bytes, plus
``4 * B * H * (Dk + Dv)`` append bytes per token. None of these kernel-level
copies is a state-management copy.

A runnable native example, including per-token allocation/copy measurements
and a multi-head fallback, is provided in
:doc:`../examples_cc/contiguous_kv_decode_example`.

Optional heterogeneous paged KV
-------------------------------

Versioned logical cache type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The kernel-independent ``PagedKVCacheTypeV1()`` declaration in ``onnx_proto``
defines the named type ``onnx_light.PagedKVCache``, version 1 (recorded as
``onnx_light.type_version = "1"``). The root is a structure containing
``blocks``, a dynamic sequence of page structures. Each page has scalar INT64
``start`` and ``length`` fields and logical FLOAT ``key`` and ``value`` tensors
with shape ``[1,1,capacity,head_size]``. Capacity and head size are not fixed
by the type; K and V may have different head sizes. The type describes the
logical structure, not a physical encoding or cache contents.

Versioned operator schema
~~~~~~~~~~~~~~~~~~~~~~~~~

``onnx_light::PagedAttention`` has an independent ``LightOpSchema`` at domain
``onnx_light``, opset 1. It takes ``Q, K, V, past`` and produces ``Y, present``.
Q/K/V and Y are FLOAT ``[1,1,L,D]`` tensors; Q/K/V use the same new-token
length, and Q/K head sizes match. ``past`` and ``present`` use the version-1
paged-cache structure. Shape inference checks known ranks and dimensions,
returns Y as ``[1,1,L,value_head_size]``, and preserves the declared cache type
for ``present`` (including a model-local struct type reference).

All attributes are optional: ``block_size=16`` and ``max_tokens=4096`` must be
positive; ``is_causal=1`` accepts only 0 or 1; ``left_window_size=-1`` means
unbounded and non-negative values bound preceding tokens. ``key_storage_type``
and ``value_storage_type`` provide default formats and accept FLOAT, INT8,
UINT8, INT4 or UINT4. Their scalar ``key_scale``/``value_scale`` default to 1
and must be positive and finite; ``key_zero_point``/``value_zero_point`` default
to 0 and must fit the selected storage range. FLOAT uses identity scale and
zero point. A kernel may override these defaults independently for K and V on
each execution; the selected format is recorded in each page rather than in
the logical cache type. These attribute value checks and data-dependent cache
checks happen at execution. The schema and shape-function registration do not
register a kernel.

Kernel implementation
~~~~~~~~~~~~~~~~~~~~~

The native ``onnx_kernels::kernel::PagedAttention`` is an opt-in consumer. It
does not change standard ONNX ``Attention`` or add another state subsystem.
Declare ``past`` and ``present`` with ``PagedKVCacheTypeV1()`` and bind
``past <- present`` in ``GraphProto.persistent_bindings``. Initialize each
request with ``PagedAttention::EmptyCache()``.

Import domain ``onnx_light`` at version 1 in the model. Unknown ranks and
symbolic Q/K/V dimensions remain supported. Cache page capacities should remain
unspecified because appended pages can have different lengths.

Register the native kernel on the context used by the state:

.. code-block:: cpp

    using namespace onnx_light;
    using namespace onnx_light::core::runtime;

    context.RegisterKernelFn(
        "onnx_light", "PagedAttention", core::symbolic::Device::kCPU,
        [](const NodeProto &node, RuntimeContext &rt) -> std::unique_ptr<KernelBase> {
          auto kernel =
              std::make_unique<onnx_kernels::kernel::PagedAttention>(rt.kernel_ctx());
          kernel->set_node(node);
          return kernel;
        });

To select storage dynamically, construct the kernel with a ``FormatSelector``.
The selector receives the current K/V tensors, the retained token count and the
attribute-derived defaults. It returns the formats for the pages appended by
that execution. For example, a policy can switch formats as the cache grows:

.. code-block:: cpp

    auto select_formats =
        [](const Tensor &, const Tensor &, int64_t past_length,
           const onnx_kernels::kernel::PagedAttention::Formats &defaults) {
          if (past_length < 1024)
            return defaults;
          return onnx_kernels::kernel::PagedAttention::Formats{
              {TensorProto::INT8, 0.01f, 0},
              {TensorProto::UINT4, 0.25f, 8}};
        };
    auto kernel = std::make_unique<onnx_kernels::kernel::PagedAttention>(
        context.kernel_ctx(), select_formats);

The node takes ``Q, K, V, past`` and returns ``Y, present``. This first consumer
accepts finite FLOAT ``[1, 1, sequence, head_size]`` tensors with equal new
Q/K/V sequence lengths and positive head sizes: multiple batches/heads,
masks and other unsupported attributes fail explicitly. Kernel instances,
execution and allocator routing use the normal runtime contracts.
Python feedback represents this cache with ``PagedCacheProto`` rather than
converting its internal sequence into a Python list.

Serialized cache values
~~~~~~~~~~~~~~~~~~~~~~~

The logical type returned by ``PagedKVCacheTypeV1()`` does not serialize a
cache value. At runtime, a cache is a ``RuntimeValue`` structure containing a
dynamic ``RuntimeSequence`` of pages. The sequence cannot be represented as one
fixed-layout ``EncodedValueProto``; an encoded value represents one tensor (or
one fixed-layout structured value), not the complete cache and its dynamic
page sequence.

``PagedCacheProto`` is the dedicated onnx-light serialized value representation,
separate from the logical ``TypeProto``. Its ``blocks`` field contains
``PagedCacheBlockProto`` messages with explicit ``start`` and ``length``.
Each block selects exactly one dense ``key`` or ``encoded_key``, and one dense
``value`` or ``encoded_value``. Dense payloads use ``TensorProto``; encoded
payloads retain ``EncodedValueProto`` layouts and parameter references.
The current kernel accepts inline dense FLOAT pages or affine INT8, UINT8, INT4
or UINT4 pages with FLOAT scales; K and V may use independent per-axis or
blocked formats. Page ranges start at zero, are contiguous and have positive
lengths no greater than their physical capacities. K/V capacities match within
each block, and each tensor's head width is consistent across blocks.

``RuntimeValue::FromPagedCache`` restores the recursive runtime value with
retained storage owners; ``ToPagedCache`` exports it without decoding pages.
Already retained dense buffers and managed encoded payloads remain shared.
Binary protobuf serialization writes their contents; parsing reconstructs
owned data. Referenced types and shared quantization parameters still belong to
the containing model, not to the standalone cache. Pass its catalogues to the
native conversion functions when needed.

``GraphProto.paged_cache_initializer`` (extension field 1002) stores named
cache defaults. Names are unique across all initializer categories. A default
that also names a graph input may be overridden by a caller; otherwise it is a
constant graph value. Runtime sessions seed these initializers, and persistent
state initialization/reset uses them when a bound input is omitted from the
initial-value map. Shape inference, model validation and ``GraphBuilder``
import/export preserve the declaration and its structured type references.
``GraphBuilder::MakePagedCacheInitializer`` adds one directly.
Non-bound graph inputs with initializer defaults may also be omitted from
current feeds; an explicit feed overrides the default for that invocation.

Python exposes both messages through ``onnx_light.onnx``. A
``PersistentValueState`` accepts and returns ``PagedCacheProto`` for these
values, including through ``values`` and ``reset``. Serializing a returned
cache and placing it in a new model's ``paged_cache_initializer`` resumes the
cache independently of the original state.

Validation rejects missing payload alternatives, negative or non-contiguous
starts, non-positive lengths, lengths larger than page capacity, inconsistent
K/V capacities or widths, non-concrete encoded payload dimensions, and external
page data. The runtime kernel additionally validates these ranges and
capacities on actual page values before attention. Load external data before
constructing the serialized cache.
This extension is supported by onnx-light binary protobuf, not standard ONNX,
ORT, or ONNX text export. Legacy graph extraction, prefixing and merging reject
cache initializers explicitly rather than silently losing them; use
``GraphBuilder`` for supported graph edits.

The cache is a named structure containing ``blocks``, a typed runtime sequence.
Historical descriptors are shared, and immutable page validation is memoized
per kernel instance. A finite attention window uses a binary search to skip
unattended historical pages; views are prepared once per invocation, not once
per query row. ``Statistics.validated_pages`` counts newly checked page metadata.
The memo is synchronized only during metadata analysis, not numerical attention.
Cache pages use independently retained storage (the I/O arena when supplied),
even when ``present`` is an intermediate value forwarded to a persistent output.
Each block is a structure with scalar INT64 ``start`` and ``length`` fields and
``key``/``value`` fields. Starts are contiguous logical token offsets; length is
the valid prefix of the block's physical token capacity. Key/value payloads are
independently owned dense tensors or affine ``EncodedValueProto`` values whose
logical shapes are ``[1, 1, capacity, head_size]``. Their affine descriptors carry
format identity, scales and zero points, independently for K, V and each block.
There are no persistent flags or new quantization layouts.

A root persistent sequence whose element type is a tensor is bridged to the
standard ``RuntimeContext::sequences()`` representation while the graph runs,
then converted back to ``RuntimeValue`` at the persistent output boundary.
Standard sequence operators such as ``SequenceInsert`` can therefore consume
and produce persistent tensor sequences. Nested sequences, including the
``blocks`` field above, remain part of their enclosing structured
``RuntimeValue``.

Persistent values also retain the shared parameter catalogue introduced by
:doc:`quantized_values`, recursively through structures and sequences. A tensor
declaration is checked against an encoded value's logical tensor type; a
structured declaration is checked against its storage type. Retaining or
validating a compact shared value checks its reference, types and byte extent
without materializing it. Borrowed views survive reset and close with both
their payload and their shared parameters.

This general runtime support does not broaden the native ``PagedAttention``
decoder: its supported page formats remain the dense and affine formats below.
Portable structured and shared encodings can be retained and forwarded by other
consumers, but this kernel rejects them explicitly rather than materializing a
whole cache or silently converting its representation.

``block_size`` bounds each block's token capacity and ``max_tokens`` bounds the
retained logical length. New chunks are converted according to
``key_storage_type``/``value_storage_type``, ``key_scale``/``value_scale`` and
``key_zero_point``/``value_zero_point``. These attributes are defaults: a
registered kernel may provide a ``FormatSelector`` that chooses independent K/V
formats from the current inputs and retained length on every execution. FLOAT,
INT8, UINT8, INT4 and UINT4 are supported for new blocks; affine append uses
scalar parameters. The selected descriptor is stored in every new page, so
successive executions may append different formats without converting prior
pages. Existing blocks may also use per-axis and blocked affine parameters with
FLOAT scales. Other scale types, external payloads and custom structured
encodings require another consumer and are rejected.
Partial blocks are sealed: an append adds new blocks
rather than rewriting the previous partial block. This can use more metadata
than filling partial blocks, but guarantees that even live aliases never force
a prefix payload copy or conversion.

Attention applies causal masking by default. ``is_causal`` and
``left_window_size`` control the visible token range. It reads only valid,
visible tokens and uses online softmax instead of concatenating K/V or allocating
a cache-length score matrix. Retention, invocation, publication and state views
share payload owners. Kernel conversion of new blocks is separate from those
zero-copy state operations.

The direct C++ call returns ``Result::statistics``:

* ``copied_bytes`` counts new stored payload bytes written by copying or
  conversion, excluding metadata and Y.
* ``dequantized_bytes`` counts decoded FLOAT bytes for visible quantized
  tokens, including repeated reads for different queries.
* ``peak_workspace_bytes`` counts peak numerical scratch, excluding output,
  retained payloads and collection metadata.

These are kernel costs, not state-forwarding copies. Native tests also check
payload/owner identity across publication and append. For example, the native
windowed fixture starts with six retained tokens, appends two tokens with
INT8 keys (width 2) and UINT4 values (width 3), and uses a left window of one:
it writes 7 new payload bytes, decodes 80 FLOAT bytes across the two queries,
and uses 24 bytes of numerical scratch. No prior block payload is copied.
The dense fixture appends three tokens with widths 3 and 2 on each call:
its 60 append bytes and 16 scratch bytes stay constant as the cache grows.

Quantization rounds ties
to even and saturates to the selected code range. For non-saturated inputs,
each affine reconstruction differs from its source by at most half its scale
(plus floating-point rounding). Attention error also depends on Q/K magnitudes
and softmax conditioning; no format alone guarantees a universal output error
bound. The numerical fixtures compare the paged consumer to dense Attention
using the reconstructed values with absolute tolerance ``1e-5`` for all four
affine storage types, separately from quantization error. The end-to-end
four-step fixture uses Q/K/V components ``+/-(0.173 * step)`` and compares to
unquantized dense Attention with these absolute output tolerances:

.. list-table::
   :header-rows: 1

   * - Storage
     - Scale
     - Zero point
     - Fixture tolerance
   * - INT8 / UINT8
     - 0.01
     - 0 / 128
     - 0.01
   * - INT4 / UINT4
     - 0.25
     - 0 / 8
     - 0.15

These fixture-specific bounds do not apply to arbitrary models or saturated
inputs.

Validation rejects malformed ranges, invalid layouts, unsupported consumers
and capacity violations. A failing or cancelled invocation cannot publish part
of a block collection. Cancellation remains a publication gate, not preemption.
Reset/close drop the state's owners; exported views and other requests retain
their blocks until their last owner is released. The bound model is never
serialized or cloned by this path.
