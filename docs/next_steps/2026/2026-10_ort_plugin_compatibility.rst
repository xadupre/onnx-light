.. _l-next-steps-ort-plugin-compatibility:

ORT-compatible custom kernels and plugin EPs without ONNX Runtime
=================================================================

:Date: 2026-10-21

**Discussion -- not implemented.** Runs independently supplied custom kernels
and plugin execution providers (EPs) in onnx-light, using their public ORT C
interfaces but **without installing, loading, linking, or executing ONNX
Runtime**. This proposal adds a compatibility host, not an ORT backend.
Only public interface declarations are reused; retain their license notices.

Scope and compatibility boundary
++++++++++++++++++++++++++++++++

Support three separately testable profiles:

* **Custom ops:** libraries exporting ``RegisterCustomOps`` and supplying
  ``OrtCustomOp`` implementations.
* **Kernel-based plugin EPs:** libraries exporting ``CreateEpFactories`` and
  ``ReleaseEpFactory``, with an ``OrtEp::GetKernelRegistry`` implementation.
* **Compiling plugin EPs:** the same factory interface, with
  ``OrtEp::GetCapability`` and ``OrtEp::Compile`` returning executable
  ``OrtNodeComputeInfo`` objects for supported partitions.

The numerical implementation belongs to the loaded library or its external
backend. Standard nodes outside EP partitions continue through onnx-light's
own dispatch. No ORT session, built-in kernel, graph optimizer, allocator,
scheduler, or fallback execution is used. A plugin may implement standard ONNX
operators itself; this does not make them ORT kernels.

An EP built against ORT's internal C++ provider interfaces is not covered.
Neither are custom kernels that invoke ORT operators through ``CreateOp`` /
``InvokeOp`` or create an ORT session internally. Vendor libraries are allowed
only if their dependency closure does not require ONNX Runtime.

Start with synchronous CPU execution and dense numeric/bool tensors. Keep
strings, sparse/structured values, device execution, shared prepacking,
EPContext persistence and graph capture as explicit extensions, not implied
compatibility. Optional and variadic tensor arguments, symbolic dimensions,
zero-element tensors and changing runtime shapes must have defined semantics
in the initial CPU profiles.

Reference ABI and support accounting
++++++++++++++++++++++++++++++++++++

The function inventory below was checked against upstream commit
``db5f5eaff7c32e09e90520374ac19b78d89094ae``:

* `onnxruntime_c_api.h
  <https://github.com/microsoft/onnxruntime/blob/db5f5eaff7c32e09e90520374ac19b78d89094ae/include/onnxruntime/core/session/onnxruntime_c_api.h>`_
  (``ORT_API_VERSION == 32``).
* `onnxruntime_ep_c_api.h
  <https://github.com/microsoft/onnxruntime/blob/db5f5eaff7c32e09e90520374ac19b78d89094ae/include/onnxruntime/core/session/onnxruntime_ep_c_api.h>`_.

This is an inventory reference, **not a claim that API 32 or every existing
binary is supported**. Pin the actual implemented ABI and fixture build
headers in the first implementation PR. A profile lists its accepted ABI
versions, platform/calling convention, tensor types, functions and callbacks.
Older layouts require separate tests; never read a newer callback field from
an older struct. Preserve the complete slot layout of each accepted
``OrtApi`` / ``OrtEpApi`` table, not a compact table of implemented functions.
Return null from ``OrtApiBase::GetApi`` for unimplemented ABI versions.

Distinguish **host functions** called by the plugin from **plugin callbacks**
called by onnx-light. All entries below are planned requirements for the
named profile, unless explicitly marked as an extension or rejection.
Each implementation PR must turn its inventory into a machine-readable
support matrix and executable contract tests.

``GetApi(version)`` negotiates a layout, not a feature set. The ORT ABI has no
general custom-kernel capability manifest: binary symbol inspection cannot
discover all indirect calls through its tables. Initially admit only audited,
tested libraries with a documented profile; do not promise arbitrary
drop-in compatibility. Unsupported status-returning slots need correctly
typed ``ORT_NOT_IMPLEMENTED`` implementations, never null pointers.
Unsupported non-status calls require a documented fail-fast policy; a fake
success, dummy device, or null buffer is not an error-reporting strategy.
Optional plugin callbacks may be absent only where the pinned ABI allows it.

Prefer plugins initialized through the supplied ``OrtApiBase`` (including
manual initialization of the header-only C++ wrappers). Also inventory the
standalone ``OrtGetApiBase`` export: a library importing it directly needs an
explicit host-export/bootstrap arrangement, tested before its static
initializers run. Do not satisfy that import by loading ``libonnxruntime``.
An existing binary with a hard dependency on an ORT shared library must be
rebuilt without that dependency; table compatibility alone cannot remove it.

Repository boundary and architecture
++++++++++++++++++++++++++++++++++++

**Implement the compatibility host in a separate repository**, provisionally
named ``onnx-light-ort-plugins``, not as a compatibility target inside
onnx-light. The name is illustrative; this proposal does not create that
repository. It depends on onnx-light and pinned public ORT headers, never on
the ORT runtime. Version and release it independently, with a tested matrix of
onnx-light versions, plugin libraries, platforms and accepted ORT ABIs.

The external repository owns plugin loading, ``OrtApi`` / ``OrtEpApi`` tables,
opaque-object adapters, custom-op and EP registration bridges, plugin-owned
resource lifetimes, and compatibility fixtures. Expose its loader and provider
selection entry points rather than implementing ORT's complete environment
and session creation APIs.

Keep only reusable, provider-neutral extension mechanisms in onnx-light:
existing kernel factories and, where required for compiling EPs, an interface
for externally prepared executable partitions with explicit inputs, outputs,
shape/memory contracts and lifecycle. No ``Ort*`` type or ORT-specific ABI
handling belongs in those interfaces. The dependency direction is strictly
external compatibility library -> onnx-light; onnx-light must not depend on
or automatically load the compatibility library.

This roadmap stays in onnx-light to document that integration boundary.
Implementation PRs belong in the external repository; any missing generic
extension points require separate, narrowly scoped onnx-light PRs. Creating
the external repository and implementing either side are outside this
documentation PR.

.. code-block:: text

   plugin .so / .dll
       -> external repository: public ORT C ABI compatibility host
       -> onnx-light: generic registration / executable partition interfaces
       -> session-owned KernelBase wrappers
       -> RuntimeContext inputs, output allocators and executor

Reuse ``RegisterKernelFn`` / context-local factories for individual kernels.
Resolve domain, operator, imported opset, type constraints and placement before
creating a kernel; a domain/name-only registration must not lose overload or
version selection. Keep existing dispatch precedence unless the user explicitly
selects an EP assignment. Do not globally overwrite unrelated kernels.
Create one plugin kernel or compiled state per resolved session node, reuse it
across runs, and pass the current context on every call.

Represent ``OrtValue``, ``OrtKernelInfo``, ``OrtKernelContext``, ``OrtGraph``,
``OrtNode`` and ``OrtValueInfo`` as host-owned adapters, not reinterpret casts
of onnx-light objects. Preserve borrowed-versus-owned return contracts.
Use borrowed buffers where safe, while retaining tensor owners and initializer
storage. Allocate outputs through the existing output-slot allocator policy
and publish only completed results. Do not permit overwriting graph inputs.

Keep library handles alive through all factories, callbacks, kernels, EPs,
compiled states, custom-op domains and plugin-owned deleters. Release compute
states and kernels before EP instances, factories and the library handle.
Handle partial registration/compilation failures with the same ownership rules.
Do not cache a construction-time ``RuntimeContext`` reference.

Host functions shared by custom ops and EPs
+++++++++++++++++++++++++++++++++++++++++++

Implement these ``OrtApi`` entries for the CPU profiles:

* **Bootstrap/errors:** ``OrtApiBase::GetApi``,
  ``OrtApiBase::GetVersionString``, ``CreateStatus``, ``GetErrorCode``,
  ``GetErrorMessage``, ``ReleaseStatus``. Identify the host as onnx-light
  compatibility support, not a real ORT installation. Translate host failures
  at status-returning C boundaries; do not let C++ exceptions escape them.
* **Dense tensors:** ``CreateTensorAsOrtValue``,
  ``CreateTensorWithDataAsOrtValue``, ``IsTensor``, ``GetTensorMutableData``,
  ``GetTensorTypeAndShape``, ``GetTensorMemoryInfo``, ``ReleaseValue``.
  External-buffer views must validate extent and retain their owner.
* **Types and dimensions:** ``GetTypeInfo``, ``GetValueType``,
  ``GetOnnxTypeFromTypeInfo``, ``CastTypeInfoToTensorInfo``,
  ``CreateTensorTypeAndShapeInfo``, ``SetTensorElementType``, ``SetDimensions``,
  ``SetSymbolicDimensions``, ``GetTensorElementType``, ``GetDimensionsCount``,
  ``GetDimensions``, ``GetSymbolicDimensions``, ``GetTensorShapeElementCount``,
  ``TensorTypeAndShape_HasShape``, ``ReleaseTensorTypeAndShapeInfo``,
  ``ReleaseTypeInfo``. Distinguish unknown shape, scalar shape and zero extent.
* **Memory descriptions:** ``CreateMemoryInfo``, ``CreateCpuMemoryInfo``,
  ``CompareMemoryInfo``, ``MemoryInfoGetName``, ``MemoryInfoGetId``,
  ``MemoryInfoGetMemType``, ``MemoryInfoGetType``, ``MemoryInfoGetDeviceType``,
  ``ReleaseMemoryInfo``.
* **Allocation:** ``GetAllocatorWithDefaultOptions``, ``AllocatorAlloc``,
  ``AllocatorFree``, ``AllocatorGetInfo`` and ``OrtAllocator`` callbacks
  ``Alloc``, ``Free``, ``Info`` and, for an accepted layout exposing it,
  ``Reserve``. Respect alignment, zero-size behavior and allocator ownership;
  do not expose unimplemented allocator extensions as usable.
* **Per-run access:** ``KernelContext_GetInputCount``,
  ``KernelContext_GetOutputCount``, ``KernelContext_GetInput``,
  ``KernelContext_GetOutput``, ``KernelContext_GetAllocator``,
  ``KernelContext_GetScratchBuffer``, ``KernelContext_ParallelFor``.
  Preserve omitted optional arguments and output lifetimes. Route parallel
  work through onnx-light's executor, with a valid serial implementation.
* **Kernel construction metadata:** ``CopyKernelInfo``, ``ReleaseKernelInfo``,
  ``KernelInfo_GetNodeName``, ``KernelInfo_GetInputCount``,
  ``KernelInfo_GetOutputCount``, ``KernelInfo_GetInputName``,
  ``KernelInfo_GetOutputName``, ``KernelInfo_GetInputTypeInfo``,
  ``KernelInfo_GetOutputTypeInfo``, ``KernelInfoGetConstantInput_tensor``,
  ``KernelInfoGetAllocator``, ``KernelInfo_GetConfigEntries``,
  ``KernelInfo_GetOperatorDomain``, ``KernelInfo_GetOperatorType``,
  ``KernelInfo_GetOperatorSinceVersion``.
* **Attributes:** ``KernelInfoGetAttribute_float``,
  ``KernelInfoGetAttribute_int64``, ``KernelInfoGetAttribute_string``,
  ``KernelInfoGetAttributeArray_float``, ``KernelInfoGetAttributeArray_int64``,
  ``KernelInfoGetAttributeArray_string``, ``KernelInfoGetAttribute_tensor``.
  Match missing-attribute, wrong-type and two-call buffer-sizing contracts.
* **Configuration/logging:** ``AddSessionConfigEntry``,
  ``HasSessionConfigEntry``, ``GetSessionConfigEntry``,
  ``GetSessionOptionsConfigEntries``, ``KernelInfo_GetLogger``,
  ``KernelContext_GetLogger``, ``Logger_GetLoggingSeverityLevel``,
  ``Logger_LogMessage``, ``CreateKeyValuePairs``, ``AddKeyValuePair``,
  ``GetKeyValue``, ``GetKeyValuePairs``, ``RemoveKeyValuePair``,
  ``ReleaseKeyValuePairs``. These operate on compatibility-host configuration
  objects; they do not create an ORT session.

Custom-op library profile
+++++++++++++++++++++++++

Host registration functions:

* Load the library, resolve and invoke its ``RegisterCustomOps`` export with
  a host-owned ``OrtSessionOptions`` adapter and ``OrtApiBase``.
* Implement ``CreateCustomOpDomain``, ``CustomOpDomain_Add``,
  ``AddCustomOpDomain``, ``ReleaseCustomOpDomain``. Commit registrations only
  when registration and descriptor validation succeed.
* Do not require ``RegisterCustomOpsLibrary``,
  ``RegisterCustomOpsLibrary_V2`` or ``RegisterCustomOpsUsingFunction`` in
  the shim: loading is initiated by the external library's API. Reject recursive
  library registration in the initial profile.

Invoke and validate these ``OrtCustomOp`` callbacks where present in the
accepted ABI:

* **Identity/signature:** ``GetName``, ``GetExecutionProviderType``,
  ``GetInputTypeCount``, ``GetInputType``, ``GetOutputTypeCount``,
  ``GetOutputType``, ``GetStartVersion``, ``GetEndVersion``. Treat a null
  provider name according to the CPU default contract, not as an arbitrary
  device fallback. Handle type-polymorphic declarations and duplicate
  domain/name registrations using validated overload resolution.
* **Argument contracts:** ``GetInputCharacteristic``,
  ``GetOutputCharacteristic``, ``GetInputMemoryType``,
  ``GetVariadicInputMinArity``, ``GetVariadicInputHomogeneity``,
  ``GetVariadicOutputMinArity``, ``GetVariadicOutputHomogeneity``.
* **Lifecycle:** prefer ``CreateKernelV2`` and ``KernelComputeV2`` for
  status-based failure propagation; also cover ``CreateKernel`` and
  ``KernelCompute`` for audited legacy libraries, plus ``KernelDestroy``.
  State explicitly how legacy callback failures are reported without
  assuming they can return an ``OrtStatus``.
* **Shape inference:** ``InferOutputShapeFn``. Implement its host dependencies
  ``ShapeInferContext_GetInputCount``, ``ShapeInferContext_GetInputTypeShape``,
  ``ShapeInferContext_GetAttribute``, ``ShapeInferContext_SetOutputTypeShape``,
  ``ReadOpAttr`` and ``ReleaseOpAttr``. Unknown output shapes may remain
  dynamic where the runtime permits them; do not invent shape information.
* **Reuse/alias declarations:** ``GetMayInplace``, ``ReleaseMayInplace``,
  ``GetAliasMap``, ``ReleaseAliasMap``. Optional in-place opportunities may
  be declined; required alias semantics must be implemented or rejected
  before execution. Always release plugin-owned index arrays through their
  matching callback, including on validation failure.

Plugin EP factory and device profile
++++++++++++++++++++++++++++++++++++

The following adds public C plugin EP support, not compatibility with internal
``IExecutionProvider`` subclasses.

Plugin exports and callbacks called by the host:

* **Exports:** ``CreateEpFactories``, ``ReleaseEpFactory``. Validate output
  counts against supplied capacities and release every returned factory.
* **``OrtEpFactory``:** ``GetName``, ``GetVendor``, ``GetVendorId``,
  ``GetVersion``, ``GetSupportedDevices``, ``CreateEp``, ``ReleaseEp``.
  Invoke optional metadata callbacks only when the accepted version and
  descriptor permit it.
* **``OrtEp``:** ``GetName``, ``GetCapability``, ``OnRunStart``, ``OnRunEnd``,
  ``IsConcurrentRunSupported``, ``Sync``, ``OnSessionInitializationEnd``,
  ``GetDefaultMemoryDevice``. Honor supported lifecycle hooks and reject
  a required unsupported mode. Plugin concurrency support never overrides
  onnx-light's own session/context serialization requirements.
* **Layout/options:** ``GetPreferredDataLayout``,
  ``ShouldConvertDataLayoutForOp`` and ``SetDynamicOptions`` require explicit
  support accounting. Initially accept the native layout and fixed options;
  reject required layout transformations or dynamic option updates rather
  than reporting that they happened.

Host functions needed for CPU device discovery and factory configuration:

* ``OrtApi::GetEpApi``; ``HardwareDevice_Type``, ``HardwareDevice_VendorId``,
  ``HardwareDevice_Vendor``, ``HardwareDevice_DeviceId``,
  ``HardwareDevice_Metadata``.
* ``EpDevice_EpName``, ``EpDevice_EpVendor``, ``EpDevice_EpMetadata``,
  ``EpDevice_EpOptions``, ``EpDevice_Device``, ``EpDevice_MemoryInfo``.
* ``OrtEpApi::CreateHardwareDevice``, ``ReleaseHardwareDevice``,
  ``CreateEpDevice``, ``ReleaseEpDevice``, ``EpDevice_AddAllocatorInfo``,
  ``GetEnvConfigEntries``. Device enumeration is owned by onnx-light;
  do not fabricate accelerators or require ``OrtEnv`` creation.
* ``OrtEpApi::MemoryInfo_GetMemoryDevice``, ``Value_GetMemoryDevice``,
  ``MemoryDevice_AreEqual``, ``MemoryDevice_GetDeviceType``,
  ``MemoryDevice_GetMemoryType``, ``MemoryDevice_GetVendorId``,
  ``MemoryDevice_GetDeviceId``.
* ``OrtApi::CreateMemoryInfo_V2``, ``MemoryInfoGetDeviceMemType``,
  ``MemoryInfoGetVendorId`` for device-aware memory descriptors.

As with custom ops, expose the external library's loading/selection APIs;
implementing
``RegisterExecutionProviderLibrary``, ``UnregisterExecutionProviderLibrary``,
``GetEpDevices`` and ``SessionOptionsAppendExecutionProvider_V2`` is not a
prerequisite. Libraries relying on those host-side convenience APIs require
an explicitly expanded profile.

Kernel-based plugin EP profile
++++++++++++++++++++++++++++++

Implement these ``OrtEpApi`` host functions:

* **Registry:** ``CreateKernelRegistry``, ``ReleaseKernelRegistry``,
  ``KernelRegistry_AddKernel``, ``EpGraphSupportInfo_LookUpKernel``,
  ``EpGraphSupportInfo_AddSingleNode``.
* **Definition construction:** ``CreateKernelDefBuilder``,
  ``ReleaseKernelDefBuilder``, ``KernelDefBuilder_SetOperatorType``,
  ``KernelDefBuilder_SetDomain``, ``KernelDefBuilder_SetSinceVersion``,
  ``KernelDefBuilder_SetExecutionProvider``,
  ``KernelDefBuilder_SetInputMemType``, ``KernelDefBuilder_SetOutputMemType``,
  ``KernelDefBuilder_AddTypeConstraint``,
  ``KernelDefBuilder_AddInputOutputAliases``,
  ``KernelDefBuilder_AddInputOutputMutableAliases``,
  ``KernelDefBuilder_Build``, ``ReleaseKernelDef``, ``GetTensorDataType``.
* **Definition queries:** ``KernelDef_GetOperatorType``,
  ``KernelDef_GetDomain``, ``KernelDef_GetSinceVersion``,
  ``KernelDef_GetExecutionProvider``, ``KernelDef_GetInputMemType``,
  ``KernelDef_GetOutputMemType``, plus ``KernelInfo_GetEp``.

Call ``OrtEp::GetKernelRegistry``, retain its borrowed registry for its specified
lifetime, select a compatible registered ``OrtKernelCreateFunc``, and call
``OrtKernelImpl::Compute`` / ``Release``. Validate type parameters against
operator schemas, not only the first input's element type.
Schema-query support comprises ``OrtEpApi::GetOpSchema``, ``ReleaseOpSchema``,
``OpSchema_GetSinceVersion``, ``OpSchema_GetNumInputs``,
``OpSchema_GetInputName``, ``OpSchema_GetInputTypeConstraint``,
``OpSchema_GetNumOutputs``, ``OpSchema_GetOutputName``,
``OpSchema_GetOutputTypeConstraint``, ``OpSchema_GetTypeConstraintCount``,
``OpSchema_GetTypeConstraint``, ``OpSchemaTypeConstraint_GetTypeParamName``,
``OpSchemaTypeConstraint_GetAllowedTypes``,
``OpSchemaTypeConstraint_GetInputIndices`` and
``OpSchemaTypeConstraint_GetOutputIndices``.

Non-sharing ``OrtKernelImpl::PrePack`` is a later optimization, not permission
to release initializer storage prematurely. Shared packing additionally needs
``OrtKernelImpl::SetSharedPrePackedWeight`` and
``OrtEpApi::SharedPrePackedWeightCache_StoreWeightData``; defer it until
ownership and cross-session sharing are specified. The initial profile must
establish that its kernels work without prepacking.

Compiling plugin EP profile
+++++++++++++++++++++++++++

Both EP profiles need graph inspection for ``GetCapability``; implement the
node/value enumeration subset before kernel-based EP assignment, then extend
it for compilation. Provide these ``OrtApi`` functions over existing
onnx-light model, node, type and shape information:

* **Graphs:** ``Graph_GetName``, ``Graph_GetModelPath``,
  ``Graph_GetOnnxIRVersion``, ``Graph_GetNumOperatorSets``,
  ``Graph_GetOperatorSets``, ``Graph_GetNumInputs``, ``Graph_GetInputs``,
  ``Graph_GetNumOutputs``, ``Graph_GetOutputs``,
  ``Graph_GetNumInitializers``, ``Graph_GetInitializers``,
  ``Graph_GetNumNodes``, ``Graph_GetNodes``, ``Graph_GetParentNode``,
  ``Graph_GetGraphView`` and ``Graph_GetModelMetadata``.
* **Model metadata:** ``ModelMetadataGetProducerName``,
  ``ModelMetadataGetGraphName``, ``ModelMetadataGetDomain``,
  ``ModelMetadataGetDescription``, ``ModelMetadataGetGraphDescription``,
  ``ModelMetadataGetVersion``, ``ModelMetadataLookupCustomMetadataMap``,
  ``ModelMetadataGetCustomMetadataMapKeys``, ``ReleaseModelMetadata``.
  Respect the ownership of the metadata handle and allocator-owned strings.
* **Nodes:** ``Node_GetId``, ``Node_GetName``, ``Node_GetOperatorType``,
  ``Node_GetDomain``, ``Node_GetSinceVersion``, ``Node_GetNumInputs``,
  ``Node_GetInputs``, ``Node_GetNumOutputs``, ``Node_GetOutputs``,
  ``Node_GetNumImplicitInputs``, ``Node_GetImplicitInputs``,
  ``Node_GetNumAttributes``, ``Node_GetAttributes``,
  ``Node_GetAttributeByName``, ``Node_GetNumSubgraphs``,
  ``Node_GetSubgraphs``, ``Node_GetGraph``, ``Node_GetEpName``.
* **Values:** ``GetValueInfoName``, ``GetValueInfoTypeInfo``,
  ``ValueInfo_GetValueProducer``, ``ValueInfo_GetValueNumConsumers``,
  ``ValueInfo_GetValueConsumers``, ``ValueInfo_GetInitializerValue``,
  ``ValueInfo_GetExternalInitializerInfo``,
  ``ValueInfo_IsRequiredGraphInput``, ``ValueInfo_IsOptionalGraphInput``,
  ``ValueInfo_IsGraphOutput``, ``ValueInfo_IsConstantInitializer``,
  ``ValueInfo_IsFromOuterScope``.
* **Attribute/initializer views:** ``OpAttr_GetName``, ``OpAttr_GetType``,
  ``OpAttr_GetTensorAttributeAsOrtValue``, ``ReadOpAttr``,
  ``ExternalInitializerInfo_GetFilePath``,
  ``ExternalInitializerInfo_GetFileOffset``,
  ``ExternalInitializerInfo_GetByteSize``. Preserve borrowed ownership and
  the existing model-relative external-data validation.

Implement ``OrtEpApi::EpGraphSupportInfo_AddNodesToFuse`` and
``NodeComputeContext_NodeName``. Invoke ``OrtEp::GetCapability``, then
``Compile`` / ``ReleaseNodeComputeInfos`` and
``OrtNodeComputeInfo::CreateState`` / ``Compute`` / ``ReleaseState``.
Keep partition graph views alive for the documented callback lifetimes.

Validate proposals before altering execution: nodes belong to the supplied
graph, partitions do not overlap, input/output ordering is deterministic,
implicit captures are retained, and fusion cannot introduce a cycle through
an unsupported node. Integrate partitions with onnx-light's execution plan;
do not introduce an ORT scheduler. Respect shape and memory metadata for
partition boundaries. Once assigned, a compile or compute error is an error,
not permission to silently fall back.

Initially support ephemeral compiled state only. Reject EPContext export,
weightless models, or requests to construct/edit models through
``GetModelEditorApi``. A plugin requiring those features needs a separate
profile with its own complete function inventory.

Device execution and other explicit extensions
++++++++++++++++++++++++++++++++++++++++++++++

Keep these dependencies visible even though they are outside the first CPU
implementation:

* **Allocators/transfers:** ``OrtEpFactory::CreateAllocator`` /
  ``ReleaseAllocator`` / ``CreateDataTransfer``, ``OrtEp::CreateAllocator``,
  ``OrtDataTransferImpl::CanCopy`` / ``CopyTensors`` / ``Release``.
  Add device-resident ``OrtValue`` handling, staging, synchronization and
  memory-budget checks before accepting a non-CPU device.
* **Streams:** ``OrtEpFactory::IsStreamAware`` /
  ``CreateSyncStreamForDevice``, ``OrtEp::CreateSyncStreamForDevice``,
  ``OrtSyncStreamImpl::GetHandle`` / ``CreateNotification`` / ``Flush`` /
  ``OnSessionRunEnd`` / ``Release``, and
  ``OrtSyncNotificationImpl::Activate`` / ``WaitOnDevice`` / ``WaitOnHost`` /
  ``Release``. Host dependencies include ``KernelContext_GetGPUComputeStream``,
  ``KernelContext_GetSyncStream``, ``KernelContext_GetResource``,
  ``OrtEpApi::SyncStream_GetImpl``, ``SyncStream_GetSyncId`` and
  ``GetSyncIdForLastWaitOnSyncStream``. A CPU placeholder must not make these
  resources appear available.
* **EP-supplied custom domains:** ``OrtEpFactory::GetNumCustomOpDomains`` /
  ``GetCustomOpDomains`` reuse the custom-op profile. Distinguish executable
  kernels from schema-only placeholders: a placeholder must be compiled by
  the EP and must never execute as a successful no-op.
* **Explicitly unsupported ORT execution helpers:** ``CreateOp``,
  ``InvokeOp``, ``CreateSession``, ``CreateSessionFromArray``, ``Run``, and
  ``OrtEpApi::CreateIfKernel`` / ``CreateLoopKernel`` / ``CreateScanKernel``.
  Existing onnx-light control flow is not an implementation of those ABI
  contracts. Do not forward them into a real ORT installation.
* **Deferred features:** string/sparse/sequence/map value APIs, shared weight
  caches, provider-specific resources, profiling, compiled-model persistence,
  external graphics resources and graph capture require additional explicit
  inventories and tests. Reject libraries whose required behavior depends
  on them until those profiles exist.

Implementation sequence and acceptance gates
++++++++++++++++++++++++++++++++++++++++++++

Steps below are implemented in the external repository. Land any prerequisite
provider-neutral extension APIs in separate onnx-light PRs before using them;
do not move ORT compatibility code into onnx-light to bypass that boundary.

1. **ABI host and loader:** pin headers and accepted layouts; add typed
   tables, error/configuration/logging objects, trusted explicit-path loading,
   ownership accounting and a support-matrix report. Test version negotiation,
   missing symbols, failed registration and teardown. The shim must build
   without any ORT library or Python package.
2. **Custom kernels:** implement the shared CPU functions and custom-op
   callbacks. Compile standalone fixture libraries using only the pinned
   headers and their own numerical code. Cover an attributed CPU op,
   type overloads, optional/variadic arguments, shape inference, dynamic
   outputs and repeated execution without repeated kernel construction.
3. **Kernel-based EP:** add factory/device objects, registry definitions,
   schema/type matching, graph enumeration and single-node capability
   assignment. A fixture EP supplies its own kernels for a small
   standard/custom operator set.
   Verify mixed native/plugin execution and reject incompatible versions,
   types, memory requirements and alias contracts.
4. **Compiling EP:** add graph adapters, partition validation and compiled-state
   lifecycle. A fixture compiles a two-node partition using its own code.
   Verify results, boundary ordering, captures, multiple partitions,
   changing shapes, initialization failures and partial cleanup.
5. **Compatibility expansion:** audit one real independent library per
   profile, record all additional API dependencies, then separately consider
   device transfers/streams and other extensions. Publish a tested
   library/platform/ABI matrix rather than a universal compatibility claim.

For each delivered profile, run end-to-end in an environment without the
``onnxruntime`` Python package or shared libraries. Inspect binary imports and
the transitive loader dependencies (including delayed loads during inference)
to establish that no ORT code is used. Compare fixture outputs to independent
expected values; use onnx-light's native kernels for non-assigned nodes.

Exercise malformed descriptors, invalid argument indices, negative runtime
dimensions, checked dimension-product/byte-size overflow, scalar/empty values,
insufficient external buffers, allocation failures, wrong devices, missing
attributes, callback errors and rejected API calls. Verify deleters run once
and no callbacks survive library unload. Test graph-input non-overwrite,
output ownership and temporary workspace lifetimes across repeated runs.

Native plugin loading executes trusted code; this is not a sandbox. Never
load a plugin automatically from model metadata. Use an explicit caller-owned
path/allowlist policy, validate resolved paths and links where containment is
required, and document that API guards cannot contain a malicious library.
Validate graph/initializer external paths with the existing model-loading
policy, and apply resource limits before exposing allocations to plugins.
