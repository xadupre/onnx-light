compute
=======

The ``compute`` helpers infer graph-level metadata generated from shape
inference and value lifetimes, and describe how a graph is executed: the
raw-buffer allocator, the individual execution actions
(:cpp:class:`ExecuteAction`), and the ordered :cpp:class:`ExecutionPlan`.

.. toctree::
    :maxdepth: 1

    compute_context
    inplace_reuse
    inplace_reuse_types
    peak_memory
    value_tags
    raw_buffer_allocator
    execute_action
    execution_plan
