// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/backend_test/io_data.h"
#include "onnx_core/backend_test/test_case.h"
#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/random.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::core::backend_test {

/**
 * Appends a *lazy* single-node :ref:`TestCase` built from ``node`` and the
 * provided typed inputs/outputs to ``registry``.
 *
 * Mirrors ``onnx_light.backend.test.case.base.expect()``. Only the inputs and
 * outputs whose name is non-empty in the node are wired into the graph. The
 * ``ModelProto`` and data set are not built at registration time; the given
 * ``node``/``inputs``/``outputs`` are captured and :func:`BuildSingleNodeCase`
 * is invoked only when the case is materialized (via ``TestCase::model`` /
 * :func:`TestCase::data_sets` / :func:`TestCase::Materialize`). The declared
 * element counts are recorded eagerly from the input/output tensors so the
 * sizing stays inspectable without materializing.
 *
 * @param node Single-node template; its ``op_type``, ``domain`` and
 *             ``attribute``s are kept.
 * @param inputs Concrete input tensors corresponding to the non-empty entries
 *               of ``node.input``.
 * @param outputs Concrete expected output tensors corresponding to the
 *                non-empty entries of ``node.output``.
 * @param name Unique test name (used both for ``TestCase.name`` and the
 *             graph name).
 * @param opset_imports Opset imports for the generated model. If empty the
 *                      caller is responsible for ensuring a default has been
 *                      applied — typically pass at least ``DefaultOpset(since_version)``.
 * @param producer_name Producer name written into the model.
 * @param registry Output registry (appended to).
 * @param tag Optional grouping tag (defaults to the node domain for
 *            non-default operator domains).
 * @param output_types Optional per-output declared type specs. When non-empty
 *                     it must contain one entry per output tensor; each output
 *                     value-info is then declared from its ``TypeSpec`` instead
 *                     of the materialized tensor type. Used to declare
 *                     ``Sequence`` / ``Map`` valued outputs whose runtime
 *                     representation is a plain ``Tensor``.
 * @throws std::invalid_argument if ``inputs.size()`` does not equal the number
 *         of non-empty entries in ``node.input`` or if ``outputs.size()`` does
 *         not equal the number of non-empty entries in ``node.output``, or if
 *         ``output_types`` is non-empty and its size does not equal
 *         ``outputs.size()``.
 */
void Expect(const NodeProto &node, const Tensors &inputs, const Tensors &outputs,
            const std::string &name, const std::vector<OpsetId> &opset_imports,
            const std::string &producer_name, std::vector<TestCase> &registry,
            const std::string &tag = "", const std::vector<TypeSpec> &output_types = {});

/**
 * Builds a single-node ``ModelProto`` and its one data set from ``node`` and
 * the provided typed inputs/outputs. This is the shared core of both
 * :func:`Expect` overloads, invoked on demand when a lazy case is
 * materialized. Only the inputs and outputs whose name is non-empty in the
 * node are wired into the graph.
 *
 * Map-typed graph inputs are supplied via ``maps``: each ``Map::name`` must
 * match a non-empty entry in ``node.input``, is declared with a
 * ``map(key_type, value_type)`` TypeProto in the graph, and is stored in
 * ``DataSet::maps`` so the runtime can retrieve it by name. The remaining
 * (tensor-typed) inputs come from ``inputs`` in positional order. The sum
 * ``inputs.size() + maps.size()`` must equal the number of non-empty entries
 * in ``node.input``.
 *
 * @throws std::invalid_argument under the same conditions as :func:`Expect`.
 */
BuiltCase BuildSingleNodeCase(const NodeProto &node, Tensors inputs, Tensors outputs,
                              const std::string &name, const std::vector<OpsetId> &opset_imports,
                              const std::string &producer_name,
                              const std::vector<TypeSpec> &output_types = {},
                              std::vector<Map> maps = {});

/**
 * Appends a *lazy* single-node :ref:`TestCase` whose inputs/outputs are
 * generated on demand by ``make_io``. Overload of :func:`Expect` for cases —
 * chiefly the ``BENCHMARK`` cases — whose (potentially very large) inputs and
 * expected outputs are too expensive to materialize at registration time.
 * ``make_io`` (which performs the input generation and kernel evaluation) is
 * invoked only when the case is materialized via ``TestCase::model`` /
 * :func:`TestCase::data_sets` / :func:`TestCase::Materialize`. ``in_counts`` /
 * ``out_counts`` record the declared element count of each input/output so the
 * sizing can be validated without running ``make_io``.
 *
 * @param registry Output registry (appended to).
 * @param node Single-node template; its ``op_type``, ``domain`` and
 *             ``attribute``s are kept. Consumed (moved).
 * @param name Unique test name.
 * @param opset_imports Opset imports for the generated model.
 * @param in_counts Declared element count of each (non-empty) input.
 * @param out_counts Declared element count of each (non-empty) output.
 * @param make_io Callable producing the concrete inputs/outputs on demand.
 * @param producer_name Producer name written into the model.
 * @param tag Optional grouping tag (defaults to the node domain for
 *            non-default operator domains).
 * @param output_types Optional per-output declared type specs (see
 *                     :func:`Expect`).
 */
void Expect(std::vector<TestCase> &registry, NodeProto node, std::string name,
            std::vector<OpsetId> opset_imports, std::vector<int64_t> in_counts,
            std::vector<int64_t> out_counts, std::function<IoData()> make_io,
            std::string producer_name = "backend-test", std::string tag = "",
            std::vector<TypeSpec> output_types = {});

/// Variant of the lazy :func:`Expect` callback which receives whether expected
/// outputs were requested. It lets benchmark collectors generate inputs without
/// invoking their output-producing oracle.
void Expect(std::vector<TestCase> &registry, NodeProto node, std::string name,
            std::vector<OpsetId> opset_imports, std::vector<int64_t> in_counts,
            std::vector<int64_t> out_counts, std::function<IoData(bool)> make_io,
            std::string producer_name = "backend-test", std::string tag = "",
            std::vector<TypeSpec> output_types = {});

/**
 * Convenience overload of the lazy :func:`Expect` that omits the element-count
 * vectors. Equivalent to calling the six-parameter lazy overload with empty
 * ``in_counts`` / ``out_counts``.  Use for small test cases where pre-declaring
 * element counts adds no value over deriving them from the materialised tensors.
 */
inline void Expect(std::vector<TestCase> &registry, NodeProto node, std::string name,
                   std::vector<OpsetId> opset_imports, std::function<IoData()> make_io,
                   std::string producer_name = "backend-test", std::string tag = "",
                   std::vector<TypeSpec> output_types = {}) {
  Expect(registry, std::move(node), std::move(name), std::move(opset_imports), {}, {},
         std::move(make_io), std::move(producer_name), std::move(tag), std::move(output_types));
}

/// Default element count for a 1-D float benchmark input of a cheap
/// element-wise operator. Sized (4M floats = 16 MiB) so a single kernel
/// evaluation processes enough data to be timed reliably (~0.1 s). Operators
/// with heavier per-element cost (transcendental, matmul, ...) pass a smaller
/// explicit size to the benchmark helpers below.
inline constexpr int64_t kBenchmarkElementwiseSize = 1 << 22;

/**
 * Appends benchmark :ref:`TestCase`s for a unary element-wise float operator.
 * ``kernel`` is any callable mapping the input ``Tensor`` to the output
 * ``Tensor`` (typically the operator's kernel functor); the expected output is
 * computed by invoking it. The generated node carries no attributes, so
 * operators whose behaviour depends on attributes should build their own
 * benchmark case instead.
 *
 * When ``with_float16`` is true (the default) a second FLOAT16 benchmark case
 * named ``name + "_float16"`` is registered alongside the FLOAT one. Operators
 * whose kernel does not support FLOAT16 must pass ``with_float16 = false``.
 *
 * A third BFLOAT16 benchmark case named ``name + "_bfloat16"`` is registered
 * by default alongside the FLOAT and FLOAT16 cases. Operators whose kernel
 * does not support BFLOAT16 must pass ``with_bfloat16 = false``.
 */
template <typename Kernel>
void ExpectBenchmarkUnaryFloat(const std::string &op_type, const Kernel &kernel,
                               const std::string &name, const OpsetId &opset,
                               std::vector<TestCase> &registry, bool with_float16 = true,
                               bool with_bfloat16 = true, int64_t size = kBenchmarkElementwiseSize,
                               uint64_t seed = 987654321ULL, const std::string &input_name = "x",
                               const std::string &output_name = "y") {
  NodeProto node;
  node.set_op_type(op_type);
  node.add_input(input_name);
  node.add_output(output_name);
  Kernel k = kernel;
  Expect(registry, std::move(node), name, {opset}, {size}, {size},
         [k, size, seed](bool generate_outputs) -> IoData {
           Tensor x = Tensor::FromFloat("", {size}, Randn<float>({size}, seed));
           IoData io{{std::move(x)}, {}};
           io.expected_outputs_generated = generate_outputs;
           if (generate_outputs) {
             io.outputs.emplace_back(k(io.inputs[0]));
           }
           return io;
         },
         "backend-test", "", {TensorTypeSpec(TensorProto::FLOAT, {size})});
  if (with_float16) {
    NodeProto node16;
    node16.set_op_type(op_type);
    node16.add_input(input_name);
    node16.add_output(output_name);
    Kernel k16 = kernel;
    Expect(registry, std::move(node16), name + "_float16", {opset}, {size}, {size},
           [k16, size, seed](bool generate_outputs) -> IoData {
             Tensor x = MakeFloat16Tensor("", {size}, Randn<float>({size}, seed));
             IoData io{{std::move(x)}, {}};
             io.expected_outputs_generated = generate_outputs;
             if (generate_outputs) {
               io.outputs.emplace_back(k16(io.inputs[0]));
             }
             return io;
           },
           "backend-test", "", {TensorTypeSpec(TensorProto::FLOAT16, {size})});
  }
  if (with_bfloat16) {
    NodeProto nodebf16;
    nodebf16.set_op_type(op_type);
    nodebf16.add_input(input_name);
    nodebf16.add_output(output_name);
    Kernel kbf16 = kernel;
    Expect(registry, std::move(nodebf16), name + "_bfloat16", {opset}, {size}, {size},
           [kbf16, size, seed](bool generate_outputs) -> IoData {
             Tensor x = MakeBfloat16Tensor("", {size}, Randn<float>({size}, seed));
             IoData io{{std::move(x)}, {}};
             io.expected_outputs_generated = generate_outputs;
             if (generate_outputs) {
               io.outputs.emplace_back(kbf16(io.inputs[0]));
             }
             return io;
           },
           "backend-test", "", {TensorTypeSpec(TensorProto::BFLOAT16, {size})});
  }
}

/**
 * Appends benchmark :ref:`TestCase`s for a binary element-wise float operator
 * with two equally-shaped 1-D inputs. ``kernel`` is any callable mapping the
 * two input ``Tensor``s to the output ``Tensor``; the expected output is
 * computed by invoking it. The generated node carries no attributes. The inputs
 * and expected output are produced lazily (see the ``make_io`` overload of
 * :func:`Expect`).
 *
 * When ``with_float16`` is true (the default) a second FLOAT16 benchmark case
 * named ``name + "_float16"`` is registered alongside the FLOAT one. Operators
 * whose kernel does not support FLOAT16 must pass ``with_float16 = false``.
 *
 * A third BFLOAT16 benchmark case named ``name + "_bfloat16"`` is registered
 * by default alongside the FLOAT and FLOAT16 cases. Operators whose kernel
 * does not support BFLOAT16 must pass ``with_bfloat16 = false``.
 */
template <typename Kernel>
void ExpectBenchmarkBinaryFloat(const std::string &op_type, const Kernel &kernel,
                                const std::string &name, const OpsetId &opset,
                                std::vector<TestCase> &registry, bool with_float16 = true,
                                bool with_bfloat16 = true, int64_t size = kBenchmarkElementwiseSize,
                                uint64_t seed = 987654321ULL) {
  NodeProto node;
  node.set_op_type(op_type);
  node.add_input("x");
  node.add_input("y");
  node.add_output("z");
  Kernel k = kernel;
  Expect(registry, std::move(node), name, {opset}, {size, size}, {size},
         [k, size, seed](bool generate_outputs) -> IoData {
           Tensor x = Tensor::FromFloat("", {size}, Randn<float>({size}, seed));
           Tensor y = Tensor::FromFloat("", {size}, Randn<float>({size}, seed + 1));
           IoData io{{std::move(x), std::move(y)}, {}};
           io.expected_outputs_generated = generate_outputs;
           if (generate_outputs) {
             io.outputs.emplace_back(k(io.inputs[0], io.inputs[1]));
           }
           return io;
         },
         "backend-test", "", {TensorTypeSpec(TensorProto::FLOAT, {size})});
  if (with_float16) {
    NodeProto node16;
    node16.set_op_type(op_type);
    node16.add_input("x");
    node16.add_input("y");
    node16.add_output("z");
    Kernel k16 = kernel;
    Expect(registry, std::move(node16), name + "_float16", {opset}, {size, size}, {size},
           [k16, size, seed](bool generate_outputs) -> IoData {
             Tensor x = MakeFloat16Tensor("", {size}, Randn<float>({size}, seed));
             Tensor y = MakeFloat16Tensor("", {size}, Randn<float>({size}, seed + 1));
             IoData io{{std::move(x), std::move(y)}, {}};
             io.expected_outputs_generated = generate_outputs;
             if (generate_outputs) {
               io.outputs.emplace_back(k16(io.inputs[0], io.inputs[1]));
             }
             return io;
           },
           "backend-test", "", {TensorTypeSpec(TensorProto::FLOAT16, {size})});
  }
  if (with_bfloat16) {
    NodeProto nodebf16;
    nodebf16.set_op_type(op_type);
    nodebf16.add_input("x");
    nodebf16.add_input("y");
    nodebf16.add_output("z");
    Kernel kbf16 = kernel;
    Expect(registry, std::move(nodebf16), name + "_bfloat16", {opset}, {size, size}, {size},
           [kbf16, size, seed](bool generate_outputs) -> IoData {
             Tensor x = MakeBfloat16Tensor("", {size}, Randn<float>({size}, seed));
             Tensor y = MakeBfloat16Tensor("", {size}, Randn<float>({size}, seed + 1));
             IoData io{{std::move(x), std::move(y)}, {}};
             io.expected_outputs_generated = generate_outputs;
             if (generate_outputs) {
               io.outputs.emplace_back(kbf16(io.inputs[0], io.inputs[1]));
             }
             return io;
           },
           "backend-test", "", {TensorTypeSpec(TensorProto::BFLOAT16, {size})});
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::core::backend_test
