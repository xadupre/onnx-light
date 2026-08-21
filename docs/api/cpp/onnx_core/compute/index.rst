compute
=======

The ``compute`` helpers infer graph-level metadata generated from shape
inference and value lifetimes, and describe how a graph is executed: the
raw-buffer allocator, the individual execution actions
(:cpp:class:`ExecuteAction`), and the ordered :cpp:class:`ExecutionPlan`.

:cpp:class:`compute::ComputeContext` is the single entry point that ties
these analyses together: :cpp:func:`compute::ComputeContext::Compute` runs
shape inference, value/node tagging, in-place reuse (with release-after and
shape-tag classification) and per-node peak memory in one call and keeps every
result alive; :cpp:func:`compute::ComputeContext::WriteToModel` pushes them
back into the model (value_info and node metadata); and
:cpp:func:`compute::ComputeContext::BuildExecutionPlan` derives the
:cpp:class:`ExecutionPlan` from that information.

.. toctree::
    :maxdepth: 1

    compute_context
    constant_info
    result_lifetime
    inplace_reuse
    inplace_reuse_types
    peak_memory
    value_tags
    raw_buffer_allocator
    execute_action
    execution_plan
    prepared_execution
    prepared_task
    resolved_model_fixture
