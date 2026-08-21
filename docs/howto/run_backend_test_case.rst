.. _l-howto-run-backend-test-case:

:html_theme.sidebar_secondary.remove:

Run a backend test case with the reference evaluator
====================================================

The backend-test catalog installed with *onnx-light* contains both models and
their reference inputs and outputs.  This walkthrough retrieves one small,
deterministic case by its exact name, renders it as ``onnx-compact`` Python,
and runs it without downloading any data.

Retrieve and display the case
-----------------------------

:func:`onnx_light.onnx.backend.get_test_case` performs an exact-name lookup.
The ``test_cc_abs`` case is compiled into the regular wheel and contains one
node and a small, fixed ``float32`` input.  The public
:func:`onnx_light.tools.translate` helper renders its model with the
``onnx-compact`` representation; :func:`onnx_light.tools.translate_header`
adds the imports needed to execute the generated expression.

.. runpython::

    from onnx_light.onnx.backend import get_test_case
    import onnx_light.tools

    case = get_test_case("test_cc_abs")
    if case is None or case.model is None or not case.data_sets:
        raise RuntimeError("The installed wheel does not contain the requested test case.")

    compact = onnx_light.tools.translate_header(
        "onnx-compact"
    ) + onnx_light.tools.translate(
        case.model,
        api="onnx-compact",
    )
    print(compact)

Execute and compare the supplied values
---------------------------------------

``data_sets`` contains NumPy values in graph-input order and expected values
in graph-output order.  :class:`onnx_light.onnx.reference.ReferenceEvaluator`
exposes those names through ``input_names``.  The checks below deliberately
test output count, shape, and dtype before applying the backend case's own
``rtol`` and ``atol`` numerical tolerances.

.. runpython::

    import numpy as np

    from onnx_light.onnx.backend import get_test_case
    from onnx_light.onnx.reference import ReferenceEvaluator

    case = get_test_case("test_cc_abs")
    if case is None or case.model is None or not case.data_sets:
        raise RuntimeError("The installed wheel does not contain the requested test case.")

    inputs, expected_outputs = case.data_sets[0]
    session = ReferenceEvaluator(case.model)
    feeds = dict(zip(session.input_names, inputs))
    outputs = session.run(None, feeds)

    if len(outputs) != len(expected_outputs):
        raise AssertionError(
            f"Expected {len(expected_outputs)} outputs, got {len(outputs)}."
        )
    for index, (actual, expected) in enumerate(zip(outputs, expected_outputs)):
        if actual.shape != expected.shape:
            raise AssertionError(
                f"Output {index} shape differs: {actual.shape} != {expected.shape}."
            )
        if actual.dtype != expected.dtype:
            raise AssertionError(
                f"Output {index} dtype differs: {actual.dtype} != {expected.dtype}."
            )
        np.testing.assert_allclose(
            actual, expected, rtol=case.rtol, atol=case.atol
        )

    print(
        f"{case.name}: {len(outputs)} output(s) match "
        f"(rtol={case.rtol}, atol={case.atol})."
    )

Enable runtime diagnostics and intermediate release
---------------------------------------------------

The evaluator accepts execution options when the session is created:

* ``verbose=1`` prints one line for every dispatched node.  Keep it at ``0``
  for normal silent execution.
* ``events_enabled=True`` records value-map changes and node dispatches in the
  session's :class:`RuntimeContext`;
  :meth:`~onnx_light.onnx.reference.ReferenceEvaluator.events` returns the
  records after a run.
* ``release_intermediates=True`` removes an intermediate value after its last
  consumer.  Setting it to ``False`` keeps intermediates until the run ends,
  which can aid debugging but increases peak memory.  This one-node case has
  no intermediate tensors to release, but the option has this effect on larger
  graphs.

The same supplied inputs and expected values can therefore exercise the
diagnostic path:

.. runpython::

    import numpy as np

    from onnx_light.onnx.backend import get_test_case
    from onnx_light.onnx.reference import ReferenceEvaluator

    case = get_test_case("test_cc_abs")
    if case is None or case.model is None or not case.data_sets:
        raise RuntimeError("The installed wheel does not contain the requested test case.")
    inputs, expected_outputs = case.data_sets[0]
    diagnostic_session = ReferenceEvaluator(
        case.model,
        verbose=1,
        events_enabled=True,
        release_intermediates=True,
    )
    diagnostic_feeds = dict(zip(diagnostic_session.input_names, inputs))
    diagnostic_outputs = diagnostic_session.run(None, diagnostic_feeds)

    for actual, expected in zip(diagnostic_outputs, expected_outputs):
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise AssertionError("The diagnostic run changed output metadata.")
        np.testing.assert_allclose(
            actual, expected, rtol=case.rtol, atol=case.atol
        )

    events = diagnostic_session.events()
    if not events:
        raise AssertionError("events_enabled=True did not record any events.")
    actions = sorted({event.as_dict()["action"] for event in events})
    print(f"Recorded {len(events)} events with actions {actions}.")

The :ref:`runtime design overview <l-design-runtime>` explains how these
Python calls map to session preparation, kernel dispatch, runtime storage, and
CPU execution.  To discover other cases, see
:ref:`l-howto-collect-backend-test-cases`.
