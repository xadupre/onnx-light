.. _l-next-steps-prepared-execution:

Parallel model initialization
=============================

:Date: 2026-08

Objective
+++++++++

The objective is to reduce model loading time. Initialization follows four
steps:

1. load the graph and initializer metadata without loading weight payloads;
2. run shape inference and determine which results are constant;
3. initialize every kernel and ask it what data preparation it requires;
4. execute the resulting initialization plan in parallel.

After these steps, the model is ready to run. Model execution itself is not
covered by this proposal.

Step 1: load the model without weights
++++++++++++++++++++++++++++++++++++++

``ParseOptions.skip_raw_data`` already loads the model structure while
skipping large ``raw_data`` fields. For external data, every initializer still
provides the information needed by the following steps:

.. code-block:: text

    name
    element type
    dimensions
    external file
    offset
    length

Loading produces a ``WeightDescriptor`` for every initializer instead of a
runtime tensor:

.. code-block:: cpp

    struct WeightDescriptor {
      std::string name;
      int32_t element_type;
      std::vector<int64_t> dimensions;
      std::string location;
      uint64_t offset;
      uint64_t length;
    };

The descriptor identifies the bytes but does not read them. Large models
should use external data because inline ``raw_data`` can be skipped but cannot
later be recovered from the metadata-only ``ModelProto`` without retaining a
reference to the original model file and its byte offsets.

Step 2: infer all available information
+++++++++++++++++++++++++++++++++++++++

Shape inference runs before any weight payload is loaded. It uses graph
inputs, initializer types and dimensions, operator attributes, and inferred
intermediate shapes.

Initialization also needs a constant-result analysis. This is distinct from
constant folding: the first analysis determines that a result is constant
without necessarily computing its bytes.

The analysis processes nodes in topological order:

1. graph initializers and outputs of ``Constant`` are constant;
2. a node output is constant when all required inputs are constant and the
   operator is deterministic and has no external state;
3. a shape result may be constant from shape information alone, even when the
   corresponding tensor payload is unavailable;
4. an output is not marked constant when it depends on a graph input, random
   state, mutable state, or an unsupported control-flow condition.

Every operator schema should expose a property such as
``CanPropagateConstant``. Operators such as ``Shape``, ``Size``, ``Gather`` on
a known shape, and simple arithmetic on constant shape tensors can additionally
provide a small-value evaluator. This evaluator computes only values required
for initialization; it does not load or fold every large constant tensor.

The result is:

.. code-block:: text

    value name -> dynamic
                constant, bytes not materialized
                constant, small value known

This information tells kernels whether an input can be prepared once and
which weight payloads will eventually be needed.

Step 3: ask kernels what they need
++++++++++++++++++++++++++++++++++

Kernel initialization must not load weights directly. Each kernel receives
the node, inferred types and shapes, and the constant-result information. It
returns a list of initialization tasks:

.. code-block:: cpp

    struct KernelInitialization {
      std::vector<InitializationTask> tasks;
    };

    KernelInitialization Kernel::Initialize(
        const NodeProto &node,
        const InferredGraph &graph,
        const WeightDescriptors &weights);

A task declares:

* its input weight ranges;
* its output prepared object;
* its dependencies on other tasks;
* the resource on which it runs: I/O, CPU, or a specific accelerator;
* an estimated amount of work and peak memory;
* whether the result is optional or required before the first inference.

A kernel that needs no preparation returns no task. A kernel may request only
loading, loading followed by prepack, or a more expensive computation.

The cost estimate should not be limited to a ``linear`` or ``quadratic`` enum.
The kernel should return concrete estimates when possible:

.. code-block:: text

    bytes_to_read
    bytes_to_write
    estimated_operations
    peak_temporary_bytes
    parallelism_limit

An asymptotic class may be reported for diagnostics, but the scheduler needs
estimated bytes and operations to choose task granularity and avoid excessive
memory use.

``Gemm`` example
^^^^^^^^^^^^^^^^

For:

.. code-block:: text

    Y = alpha * op(A) * op(B) + beta * C

if ``B`` is constant, ``Gemm`` may request a prepack task for ``B``. The task
includes ``transB`` because the required representation is that of ``op(B)``,
not always that of the stored tensor.

The model loader does not transpose ``B``. The selected kernel decides whether
to:

* use the original layout and pass ``transB`` to the GEMM library;
* transpose while packing;
* build another backend-specific packed representation.

The original bytes remain unchanged. If the same initializer is consumed by
two nodes with different ``transB`` values, the kernels may request two
different prepared objects.

For the usual dense case, reading and transposing or packing ``B`` is linear
in the number of elements. A kernel that requests a quadratic preprocessing
step must declare that cost and its temporary memory explicitly.

Step 4: execute the initialization plan
+++++++++++++++++++++++++++++++++++++++

All kernel requests are merged into one dependency graph. Identical requests
share one task:

.. code-block:: text

    read W0 ----> prepack W0 ----> kernel 0 ready
       |
       +--------> prepack W0' ---> kernel 7 ready

    read W1 ----> copy to CUDA ---> kernel 1 ready

    compute small constant -------> kernel 2 ready

The scheduler executes ready tasks in parallel. It uses separate queues for:

* file reads;
* CPU transformations;
* each accelerator.

It also enforces a global in-flight memory budget. Without this budget,
parallel reads and prepacks may temporarily allocate both the source and
prepared forms of every weight.

Loading and prepack overlap naturally: ``prepack W0`` starts when ``W0`` is
available while another I/O worker reads ``W1``. A task releases its source
buffer when no later task needs it. External-data mappings may instead remain
available as the portable backing store.

Kernel and device
+++++++++++++++++

A kernel implementation should remain attached to one **execution device**.
A CUDA kernel produces CUDA outputs and participates in a CUDA execution
schedule. Treating it simultaneously as a CPU and CUDA kernel would make
placement, output ownership, and transfer insertion ambiguous.

However, its **initialization tasks** may use several resources. For example:

.. code-block:: text

    CUDA Gemm kernel
      execution device: CUDA

      initialization:
        read B                 -> I/O
        transpose/pack B       -> CPU
        upload packed B        -> CUDA

or:

.. code-block:: text

    CUDA Gemm kernel
      execution device: CUDA

      initialization:
        read B                 -> I/O
        upload B               -> CUDA
        pack B                 -> CUDA

The CUDA kernel may therefore use the CPU during initialization without
becoming a CPU execution kernel. The kernel chooses between CPU packing and
CUDA packing according to the available implementation and reports the
corresponding task graph.

If an operator can execute entirely on either CPU or CUDA, it has two kernel
implementations. The session selects one execution kernel for the current
placement:

.. code-block:: text

    GemmCpu   -> execution device CPU  -> CPU packed weight
    GemmCuda  -> execution device CUDA -> CUDA packed weight

They may share the same source ``WeightDescriptor`` and the same read task, but
their prepared objects are distinct.

Multiple devices
++++++++++++++++

The first implementation should assign one execution device to every node
before kernel initialization. Kernel selection then produces the initialization
tasks for that fixed placement. This keeps the first version simple while
already allowing all I/O, CPU preparation, and accelerator preparation to
overlap.

Supporting a placement that changes later requires retaining one prepared
variant per ``(node, execution device)``. Changing placement selects another
kernel and may trigger its missing initialization tasks. This is an extension
of the same plan, not a reason to make one kernel multi-device.

Offloading between inference iterations is outside the initial loading plan.
It should be implemented as a residency policy over already defined CPU and
accelerator kernel variants. The portable source weight remains addressable,
and prepared forms may be cached or evicted independently. This can be added
after fixed-placement parallel initialization works.

Benchmark
+++++++++

The benchmark must start from a valid serialized model and must not modify its
graph or synthesize weights. The current Qwen3-like backend fixture contains
metadata-only initializers, so materializing random weights, inlining
functions, and deleting ``value_info`` changes the workload. It may remain a
session microbenchmark, but it is not the loading benchmark for this work.

The loading benchmark should use a deterministic model with real external
weights and measure:

* metadata-only parsing;
* shape and constant-result inference;
* kernel initialization and task-graph construction;
* weight reads;
* prepack and device transfers;
* total time until all required kernels are ready;
* peak memory and maximum bytes in flight.

It should compare sequential and parallel execution of exactly the same
initialization plan.

Implementation order
++++++++++++++++++++

1. Add a valid external-data model benchmark that is never rewritten by the
   benchmark.
2. Build ``WeightDescriptor`` objects while parsing with
   ``skip_raw_data=true``.
3. Add constant-result propagation without payload materialization.
4. Add the kernel initialization query and an empty default implementation.
5. Implement initialization tasks for one CPU ``Gemm`` with a constant ``B``,
   including both ``transB`` values.
6. Merge the kernel tasks into a dependency graph and execute it sequentially.
7. Add the bounded I/O and CPU task queues, then compare the same plan in
   sequential and parallel modes.
8. Add one CUDA ``Gemm`` whose initialization can choose CPU-pack-plus-copy or
   CUDA-side pack.
9. Add alternative execution-device variants and weight residency only after
   fixed-placement initialization is complete.
