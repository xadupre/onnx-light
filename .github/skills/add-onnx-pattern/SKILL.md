---
name: add-onnx-pattern
description: Adds or ports an ONNX graph-rewriting pattern. Use when implementing a pattern under onnx_light/onnx_extensions/patterns.
---

# Add an ONNX pattern

Follow every step when adding or porting a graph-rewriting pattern.

1. Establish the semantics.
   - Read the authoritative implementation and tests when porting a pattern.
   - Record the exact supported operators, domains, opsets, data types, constants,
     graph-use constraints, and replacement graph.
   - Preserve upstream guards when the local graph representation supports them.
     Explain any unsupported metadata or intentional safety restriction.

2. Implement the pattern.
   - Place the declaration and implementation in the matching category under
     `onnx_light/onnx_extensions/patterns/`.
   - Derive from `core::builder::PatternOptimization`, give it a stable intrinsic name,
     and implement `FastOpType`, `Match`, and `Apply`.
   - Restrict matching to the intended domain and operator forms.
   - Reject graph outputs, shared values, subgraph captures, unknown metadata, or
     unsafe broadcasting whenever removing a matched node could change behavior.
   - Re-run `Match` from `Apply` and reject inconsistent direct calls with
     `core::builder::BuilderError`.
   - Preserve externally visible output names and relevant node metadata.
   - Document the rewrite in the header with labeled `Before` and `After` diagrams
     using aligned extended-Unicode square boxes and arrows.

3. Register every public entry point.
   - Add the pattern factory to
     `onnx_light/onnx_extensions/patterns/dispatch_table.cc`.
   - Add the matching nanobind class in `onnx_light/onnx_py/_onnxpy_patterns.cc`.
   - Export the class and include it in `__all__` in
     `onnx_light/onnx_core/optimization.py`.
   - Keep the registry name, intrinsic name, and Python class name consistent with
     neighboring patterns.

4. Add focused tests.
   - Add C++ tests in the corresponding
     `unittests/cc/onnx_extensions/patterns/` test file.
   - Cover each supported input ordering or data type that changes matching logic.
   - Cover every important rejection guard, especially invalid constants, shared
     intermediates, graph outputs, domains, shapes, and types.
   - Assert the replacement operators, input order, and preserved output names.
   - Add an optimizer-level test when direct `Match` and `Apply` tests do not verify
     cleanup, registration, or convergence behavior.

5. Validate the change.
   - Run `clang-format -i` on every modified `.h`, `.hpp`, `.cc`, and `.cpp` file.
   - Run `black . && ruff check .`.
   - If `build/` is absent, configure it with:
     `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DONNX_LIGHT_BUILD_PYTHON=OFF
     -DONNX_LIGHT_BUILD_KERNELS=OFF -DONNX_LIGHT_BUILD_TESTS=ON`.
   - Build the relevant C++ test target:
     `cmake --build build --target test_onnx_light -j2`.
   - Run the focused tests with
     `./build/test_onnx_light --gtest_filter='<PatternName>.*'`.
   - If Python bindings changed and Python behavior is tested, rebuild the extension
     from the working tree with `python setup.py build_ext --inplace` before running
     those tests.
   - Review the final diff to confirm registration, bindings, tests, and documentation
     are all present and no unrelated files changed.
