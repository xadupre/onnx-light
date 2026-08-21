.. _l-next-steps-fast-loading-sequence:

Fast-loading implementation sequence
====================================

:Date: 2026-08

Large-model startup follows one explicit four-document sequence:

1. fix existing parser, external-data, and initializer-materialization defects
   in :ref:`l-next-steps-model-loading-bug-fixes`;
2. implement :ref:`l-next-steps-prepared-execution`;
3. complete the ownership-aware cross-repository work in
   :ref:`l-next-steps-model-loading`;
4. connect adaptive I/O, model resolution, prepared tensors, and first-token
   overlap in :ref:`l-next-steps-native-fast-loading-completion`.

Parallel-for profiling may proceed alongside step 1, but its executor
instrumentation must be stable before step 2 begins.

Assignable issue sequence
+++++++++++++++++++++++++

.. list-table::
    :header-rows: 1
    :widths: 12 24 64

    * - Step
      - Issues
      - Order
    * - 1. Bug fixes
      - #4608--#4610
      - `#4608 <https://github.com/xadupre/onnx-light/issues/4608>`_ ->
        `#4609 <https://github.com/xadupre/onnx-light/issues/4609>`_ ->
        `#4610 <https://github.com/xadupre/onnx-light/issues/4610>`_
    * - 2. Prepared execution
      - #4613--#4617
      - `#4613 <https://github.com/xadupre/onnx-light/issues/4613>`_ ->
        `#4614 <https://github.com/xadupre/onnx-light/issues/4614>`_ ->
        `#4615 <https://github.com/xadupre/onnx-light/issues/4615>`_ ->
        `#4616 <https://github.com/xadupre/onnx-light/issues/4616>`_ ->
        `#4617 <https://github.com/xadupre/onnx-light/issues/4617>`_
    * - 3. onnxruntime
      - #4611--#4612
      - `#4611 <https://github.com/xadupre/onnx-light/issues/4611>`_ ->
        `#4612 <https://github.com/xadupre/onnx-light/issues/4612>`_. #4612 is
        coordination for a PR in ``microsoft/onnxruntime`` and must not be
        assigned to an agent working only in ``xadupre/onnx-light``.
    * - 4. Native completion
      - #4618--#4623
      - `#4618 <https://github.com/xadupre/onnx-light/issues/4618>`_ ->
        `#4619 <https://github.com/xadupre/onnx-light/issues/4619>`_ ->
        `#4620 <https://github.com/xadupre/onnx-light/issues/4620>`_ ->
        `#4621 <https://github.com/xadupre/onnx-light/issues/4621>`_ ->
        `#4622 <https://github.com/xadupre/onnx-light/issues/4622>`_ ->
        `#4623 <https://github.com/xadupre/onnx-light/issues/4623>`_

Plans
+++++

.. toctree::
    :maxdepth: 1

    2026-08_model_loading_bug_fixes
    2026-08_prepared_execution
    2026-08_onnxruntime_fast_model_loading
    2026-08_native_fast_loading_completion
