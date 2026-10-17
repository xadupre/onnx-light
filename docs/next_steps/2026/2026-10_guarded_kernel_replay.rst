.. _l-next-steps-guarded-kernel-replay:

Captured execution plan: before and after
=========================================

:Date: 2026-10-17

**Discussion -- not implemented.** Proposed APIs below are illustrative C++,
not existing declarations.

ExecutionPlan: before / after
+++++++++++++++++++++++++++++

Before
^^^^^^

``RuntimeSession`` interprets ``ExecutionPlan`` actions and invokes the resolved
kernel's ordinary ``Run``. Each kernel can repeat shape planning, attribute
decoding, dtype dispatch and validation before computing.

After
^^^^^

Lower the existing schedule into a session-owned **CapturedExecutionPlan**:
a list of named functions with their numerical implementations and parameters
already selected.

.. code-block:: cpp

   struct CapturedCall {
     std::string name;
     std::function<void(ExecutionFrame&)> invoke;
   };

   class CapturedExecutionPlan {
   public:
     ExecutionFrame Bind(RuntimeContext& rt) const;
     void Run(ExecutionFrame& frame) const;
     std::string DebugString() const;

   private:
     std::vector<CapturedCall> calls_;
   };

.. code-block:: cpp

   auto frame = plan.Bind(rt);
   plan.Run(frame);

``Bind`` checks the external input signature, accessible buffers and ownership
once at the plan boundary. ``Run`` calls ``invoke(frame)`` in order: no per-node
shape guards, capture status, implementation selection or recapture protocol.
It retains the required allocator checks, inspection and data-dependent safety
checks; eliminating shape planning is not a global validation bypass.

Keep ``ExecutionPlan`` as the structural source for ordering, slots and
lifetimes. Do not introduce another graph analysis, scheduler or prepared-weight
store.

Kernel API
++++++++++

``Capture`` returns **one named callable**, always using the same return type:

.. code-block:: cpp

   class Add : public KernelBase {
   public:
     void Run(RuntimeContext& rt) override;
     CapturedCall Capture(CaptureContext& context);
   };

During capture, a kernel selects its numerical body and fixes output shapes,
strides, numerical parameters and workspace requirements. It declares its
allocation needs through ``CaptureContext`` and returns the numerical call.
The plan builder inserts allocation/publication/release calls around it.
Capture itself performs no numerical execution or physical buffer allocation.

``Run`` and checked standalone ``operator()`` APIs remain available.
``KernelBase::Prepare`` keeps its existing immutable-input preparation role.
Discover the optional capture function through a versioned descriptor associated
with the exact resolved factory, without changing ``KernelBase``'s vtable.
Negotiate version, size, ABI and ownership with external libraries.

Kernels without a specialized implementation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Return the same ``CapturedCall`` containing a wrapper around ordinary ``Run``.
Use an explicit name such as ``onnx-light::NonZero::Run``. There is no separate
result type, success status or per-call branch to interpret.

This wrapper declares no additional output/workspace allocations: ``Run``
already manages them. It restores the node's dispatch context, slots and
allocator roles, then exposes its current outputs through the frame.
Determine this path before registering capture-specific allocation requirements.
Invalid inputs raise an error rather than selecting this path.

If such a kernel has data-dependent output shapes, consumers without
independently established shape contracts also use their ordinary ``Run``.
Independent static regions can remain specialized. Do not reintroduce shape
guards or partial-graph retries inside captured replay.

CaptureContext and ExecutionFrame
+++++++++++++++++++++++++++++++++

.. code-block:: cpp

   class CaptureContext {
   public:
     bool HasInput(size_t index) const;
     uint32_t InputSlot(size_t index) const;
     uint32_t OutputSlot(size_t index) const;
     int32_t InputType(size_t index) const;
     const Shape& InputShape(size_t index) const;
     void DefineOutput(size_t index, int32_t dtype, const Shape& shape);
     uint32_t RequestScratch(size_t bytes, size_t alignment);
   };

   class ExecutionFrame {
   public:
     const Shape& ShapeAt(uint32_t slot) const;
     const Tensor& TensorAt(uint32_t slot) const;
     const RuntimeValue& ValueAt(uint32_t slot) const;
     Tensor& AllocateOutput(uint32_t slot);
     Tensor& OutputTensor(uint32_t slot);
     void AllocateScratch(uint32_t workspace, size_t bytes, size_t alignment);
     std::span<std::byte> Scratch(uint32_t workspace);
     void ReleaseScratch(uint32_t workspace);
     void PublishOutputs();
     void Release(uint32_t slot);
     RuntimeContext& runtime();
   };

``CaptureContext`` uses concrete metadata available during preparation, not
mutable data from a first execution. Attributes, tuning and immutable prepared
objects come from the resolved kernel. ``DefineOutput`` and ``RequestScratch``
declare requirements only; positions are resolved to invocation-wide slots.

``Bind`` initializes frame metadata from the prepared descriptors and binds
current feeds. A value slot identifies a logical value, not a physical buffer
or captured address. Allocation calls obtain execution/I/O buffers from the
existing allocator and install their leases in the frame.

After allocation, a numerical call resolves current pointers:

.. code-block:: cpp

   const float* x = frame.TensorAt(x_slot).As<float>();
   const float* y = frame.TensorAt(y_slot).As<float>();
   float* output = frame.OutputTensor(output_slot).As<float>();
   ComputeAddFloat(x, y, output, parameters);

``parameters`` is immutable capture-time metadata for the prepared signature.
The checked API uses the same numerical body. New input addresses do not
require recapture. Shapes, logical byte counts and capacities remain distinct.

Workspace IDs are separate from value slots. Named allocation/release calls
cover their lifetimes, and released storage can be reused where lifetimes do
not overlap. References and leases remain alive through worker completion;
exception cleanup uses RAII. Retained outputs remain valid, imported storage
remains caller-owned, and graph inputs are never overwritten or released as
intermediates. In-place reuse requires the existing last-use/aliasing proof.

The session retains referenced kernels/prepared objects until captured calls
are destroyed. Closures do not store invocation pointers or mutable bindings;
there is no per-kernel binding cleanup operation.

Named calls and inspection
++++++++++++++++++++++++++

Every call has an owned, non-empty name supplied during construction. Numerical
names identify the selected backend, routine, dtype signature, ISA and relevant
algorithm/layout variant, not merely the ONNX operator.

``DebugString()`` prints execution order, source nodes and input/output slots,
including allocation and release calls. A wrapper invoking multiple selected
subkernels lists them in its description. Ordinary ``Run`` wrappers are named
explicitly too. Do not infer names from lambda types or function addresses,
format strings on each invocation, or execute the plan merely to inspect it.

Changing dimensions
+++++++++++++++++++

The initial plan is prepared for concrete shapes. A signature mismatch is
handled before ``Run`` by selecting/preparing a compatible plan, never by
retrying a partially executed graph. Preparation recomputes only affected
descriptors, calls and dependent consumers; it does not rebuild the model or
``ShapesContext``. Start with one active plan, not an unbounded shape cache.

A later parametric plan can update explicit dimension/stride/workspace
parameters before replay using derived formulas. This must not become generic
shape inference or per-kernel validation inside the call list.

Changing contents of indices, axes, shape tensors, scales or page tables are
not covered by an unchanged input shape. Retain necessary data-dependent
validation before unsafe access, or use ordinary ``Run`` where a suitable
specialized implementation is unavailable. Bounds, checked byte arithmetic,
alignment, capacity and legal aliasing still need established proofs.

Concrete CapturedExecutionPlan
++++++++++++++++++++++++++++++

Use FLOAT32 ``[B, heads, S, D] = [1, 8, 128, 64]``, normalization on the last
axis, and rank-4 attention with ``Q = K = V = N1``. Omit projections, masks,
KV caches and optional outputs:

.. code-block:: text

   N1 = LayerNormalization(X)
   A  = Attention(N1, N1, N1)
   R  = Add(X, A)
   Y  = LayerNormalization(R)

Activation shapes and parameters are computed during capture, not by calls in
this plan. Each activation occupies 256 KiB. Slots are ``X=0, N1=1, A=2, R=3,
Y=4``; scale/bias inputs are omitted below.

.. code-block:: text

   00 allocate/value=1/role=execution
   01 allocate/workspace=0/bytes=8192
   02 onnx-light::LayerNormalization<float>/axis-last
        input=0, output=1, workspace=0
   03 publish-verify/value=1
   04 release/workspace=0

   05 allocate/value=2/role=execution
   06 allocate/workspace=1/bytes=12288
   07 onnx-light::Attention<float>/portable-rowwise
        Q=1, K=1, V=1, output=2, workspace=1
   08 publish-verify/value=2
   09 release/workspace=1
   10 release/value=1

   11 allocate/value=3/role=execution
   12 onnx-light::Add<float>/same-shape
        inputs=[0,2], output=3
   13 publish-verify/value=3
   14 release/value=2

   15 allocate/value=4/role=io
   16 allocate/workspace=2/bytes=8192
   17 onnx-light::LayerNormalization<float>/axis-last
        input=3, output=4, workspace=2
   18 publish-verify/value=4
   19 release/workspace=2
   20 release/value=3

These workspace sizes reflect the current portable strategy: two FLOAT32
normalization statistics per row (8 KiB), and three DOUBLE attention row buffers
for four active scratch lanes (12 KiB). Capture must establish that lane count
from the selected execution policy, not assume it equals the thread count.

Buffer storage formerly used for value 1 can be reused for value 3 after call
10. Workspace lifetimes do not overlap. Value 0 remains unchanged for the
residual Add; value 4 retains its I/O owner after intermediates are released.

Kernel migration and onnx-light-cpu
+++++++++++++++++++++++++++++++++++

All kernels follow the same API: return one specialized call or one ordinary
``Run`` wrapper. Unary/binary kernels prepare type/broadcast parameters;
matrix, convolution and pooling kernels prepare algorithms and workspaces;
reductions and normalization prepare axes and extents; indexing and stateful
kernels retain their required data checks. Control-flow, structured,
data-dependent-output and training paths may initially return ordinary calls.

CPU factories use the same API and reuse existing ``GemmPlan``, ``GemmHalfPlan``,
``MatMulPlan``, binary broadcast/SIMD plans and immutable ``MatMulNBits``
preparations. Preserve tuning, ISA policy, NAIVE/OPTIMIZED selection and factory
precedence. Do not route specialized calls through the legacy
``detail::AsSessionKernel`` signature-key/cache adapter.

Pull request breakdown
+++++++++++++++++++++

Each PR preserves ordinary execution and keeps captured replay opt-in. The
numbers below identify proposed increments, not existing GitHub PRs. Merge the
onnx-light API before its onnx-light-cpu consumers; no PR requires simultaneous
unreleased changes in both repositories.

1. **onnx-light: plan infrastructure and ordinary calls.** Add the named-call
   list, frame, versioned capture descriptor and boundary binding. Initially
   wrap every kernel's ordinary ``Run``; reuse existing scheduling, allocations
   and lifetimes. Test execution parity, inspection without execution, changed
   feed addresses and retained outputs. Incompatible signatures use ordinary
   execution before any replay effects, not a partial replay retry.
2. **onnx-light: portable Add and Mul.** Depends on PR 1. Introduce the first
   specialized calls and capture-time output/broadcast descriptors, with named
   allocation, publication and release calls. Test scalar and broadcast inputs,
   empty tensors, changing buffers, intermediate reuse and invalid inputs.
   Keep graph inputs immutable and reject invalid sizes before memory access.
3. **onnx-light: portable MatMul.** Depends on PR 2's allocation/frame path.
   Capture rank/batch handling, numerical dispatch and workspace requirements
   using existing numerical bodies. Test supported rank combinations, zero
   extents, workspace reuse and cleanup on failure, including worker completion.
   Compare a mixed Add/MatMul graph with ordinary execution.
4. **onnx-light-cpu: capture integration and binary kernels.** Depends on the
   released API from PR 2. Negotiate the descriptor at the resolved factory and
   capture existing binary broadcast/SIMD plans without the legacy adapter.
   Test unsupported descriptor versions, ordinary wrappers, factory precedence,
   tuning and ISA selection with both NAIVE and OPTIMIZED policies.
5. **onnx-light-cpu: Gemm and MatMul.** Depends on PR 4 and the released
   workspace contract from PR 3. Reuse ``GemmPlan``, ``GemmHalfPlan`` and
   ``MatMulPlan``. Test transpose/batch variants, supported dtypes, immutable
   prepared weights, changing input buffers and scratch lifetimes.
6. **onnx-light: normalization and Attention.** Depends on PR 3. Capture axes,
   extents and the selected scratch-lane policy. Exercise the complete
   normalization/Attention/Add/normalization example above, including numerical
   parity, workspace reuse and input/output ownership. Keep unsupported masks,
   caches and data-dependent cases on explicit ordinary calls.
7. **onnx-light: signature changes and targeted repreparation.** Depends on
   PRs 2 and 3; PR 6 supplies an additional integration case when available.
   Replace the initial signature-mismatch path with preparation of affected
   descriptors and consumers before replay. Test alternating dimensions,
   unchanged shapes with changing control-tensor contents, and mixed regions
   after data-dependent outputs. Never retry partially executed stateful calls
   or rebuild the whole graph/``ShapesContext``.
8. **Both repositories: measurement and rollout decision.** Depends on PRs
   3--7. Add reproducible comparisons for preparation, binding, replay and
   end-to-end latency, using identical numerical bodies and representative
   small/large kernels. Measure split versus grouped allocation/publication
   calls. Any default-enablement change is a separate, evidence-backed PR in
   each repository, not an automatic consequence of completing this list.

After the first kernels, migrate remaining families in separate bounded PRs:
unary/reduction, convolution/pooling, indexing, and CPU quantized kernels such
as ``MatMulNBits``. Each depends only on the merged infrastructure it uses and
includes parity, invalid-input and lifetime cases for that family. Stateful,
control-flow, structured and training kernels retain named ordinary calls
until a dedicated PR establishes their capture contract; completing all
families is not a prerequisite for measuring the initial implementation.

No latency gain or validation-overhead percentage has been measured for this
proposal. Separate simple checks from repeated planning and metadata allocation
when profiling. Compare identical numerical bodies and account for the extra
indirect calls used for allocation, publication and release; splitting those
operations must not erase the benefit.

Keep the existing serialized session/context contract. Enable by default only
after correctness, compatibility and repeatable performance gains are established.
