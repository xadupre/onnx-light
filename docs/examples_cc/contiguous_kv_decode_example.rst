.. _l-cpp-contiguous-kv-decode-example:

Contiguous KV decode on CPU
==========================

This example builds an immutable Attention model whose graph declares K/V
feedback, then runs twenty single-token decode steps. Zero Q/K tensors make
attention uniform, so every result is checked against the mean of the values
seen so far. The example compares one KV head, eligible for contiguous tail
reuse, with two KV heads, which require dense concatenation.

The initial capacity is four tokens, so tokens 5, 9 and 17 exercise growth.
CSV output reports latency, kernel allocation requests and capacity bytes,
prefix/append copy bytes, buffer reuse counts, and I/O-arena live/peak bytes.
Each row also verifies that the state and returned K/V outputs share pointers:
state forwarding does not duplicate payloads. The per-token kernel counters
exclude feed construction, Attention arithmetic, and score/output workspace.

For one KV head of width four, nongrowing reuse steps report zero allocations,
zero prefix-copy bytes and 32 append-copy bytes. The two-head fallback reports
two allocations and ``64 * (token - 1)`` prefix-copy bytes per step.

Build against an installed native onnx-light tree:

.. code-block:: bash

    cmake -S examples/contiguous_kv_decode -B build-contiguous-kv-decode \
          -DCMAKE_PREFIX_PATH=/path/to/onnx-light-install
    cmake --build build-contiguous-kv-decode
    ./build-contiguous-kv-decode/contiguous_kv_decode

Release each returned output before the next call to allow exclusive reuse.
Holding an output or state view is supported, but forces a fresh allocation
when that buffer would otherwise be reused. See
:ref:`l-howto-persistent-feedback` for the ownership, valid-length and
cancellation contracts.

.. literalinclude:: ../../examples/contiguous_kv_decode/main.cc
    :language: cpp
    :linenos:
