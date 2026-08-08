/**
 * main.cc — Standalone example: implement a brand-new C++ kernel *class* for
 * an existing ONNX operator, register it into onnx-light's shared kernel
 * dispatch table, run a model that uses that operator and verify the new
 * kernel is the one actually executed.
 *
 * This is the scenario implemented by the companion ``onnx-light-cpu``
 * project (`https://github.com/xadupre/onnx-light-cpu`), which ships
 * SIMD-accelerated ``Abs`` / ``Exp`` / ``Log`` / ``Gemm`` / ``Not`` kernels as
 * :cpp:class:`onnx_light::core::runtime::KernelBase` subclasses and installs
 * them into onnx-light's dispatch table so *any* model using those operators
 * runs the optimized kernels instead of the built-in ones. Here we implement a
 * single, self-contained ``Abs`` replacement to keep the example short.
 *
 * The moving parts are:
 *
 *  1. ``ExampleAbsKernel`` — a ``KernelBase`` subclass computing the
 *     element-wise absolute value of a FLOAT tensor. Like every built-in
 *     kernel it exposes a ``static constexpr const char *name`` identifier
 *     (``"example:CPU:ai.onnx:Abs"``) following the
 *     ``"<library>:<device>:<domain>:<op_type>"`` convention used by
 *     onnx-light's own kernel classes.
 *  2. ``RegisterExampleAbsKernel`` — installs a factory for the kernel into
 *     the shared table via
 *     :cpp:func:`onnx_light::core::runtime::RegisterKernelFn` for the CPU
 *     device and the default ONNX domain, overriding the built-in ``Abs``.
 *  3. ``main`` — registers the built-in kernels first, then the override,
 *     builds a one-node ``Abs`` graph, runs it through a
 *     :cpp:class:`onnx_light::core::runtime::RuntimeSession` and checks both
 *     that the output equals ``|x|`` and that ``ExampleAbsKernel`` — not the
 *     built-in — produced it (a run counter is bumped on every dispatch).
 *
 * See CMakeLists.txt for build instructions.
 */

#include "onnx_core/compute/execution_plan.h"
#include "onnx_core/runtime/kernel_context.h"
#include "onnx_core/runtime/kernel_dispatch_table.h"
#include "onnx_core/runtime/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_core/runtime/runtime_session.h"
#include "onnx_core/runtime/simple_tensor.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_extensions/kernels/kernel_dispatch_table.h"
#include "onnx_proto/onnx.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace rt = ONNX_LIGHT_NAMESPACE::core::runtime;
namespace sym = ONNX_LIGHT_NAMESPACE::core::symbolic;
using ONNX_LIGHT_NAMESPACE::GraphProto;
using ONNX_LIGHT_NAMESPACE::NodeProto;
using ONNX_LIGHT_NAMESPACE::ValueInfoProto;

namespace example {

// Counts how many times ExampleAbsKernel::Run was invoked, so the example can
// prove the custom kernel — and not the built-in one — executed the node.
int g_example_abs_run_count = 0;

/// A brand-new kernel *class* for the existing ONNX ``Abs`` operator.
///
/// It derives from onnx-light's :cpp:class:`KernelBase` exactly like a
/// built-in kernel, so the dispatch table constructs it once per node and
/// calls :cpp:func:`Run` on every execution.
class ExampleAbsKernel : public rt::KernelBase {
public:
  // Every onnx-light kernel class exposes a unique, human-readable identifier
  // following the "<library>:<device>:<domain>:<op_type>" convention (see the
  // built-in kernels, e.g. "onnx_kernels:CPU:ai.onnx:Abs"). Custom kernels use
  // their own library prefix so their name never collides with a built-in one.
  static constexpr const char *name = "example:CPU:ai.onnx:Abs";

  using rt::KernelBase::KernelBase;

  void Run(rt::RuntimeContext &ctx) override {
    const NodeProto &node = *node_;
    rt::RequireInputCount(node, 1);
    rt::RequireOutputCount(node, 1);
    const rt::Tensor &x = rt::GetInput(node, 0, ctx.tensors());
    if (x.data_type != static_cast<int32_t>(rt::DataType::FLOAT)) {
      throw std::invalid_argument("example::ExampleAbsKernel only supports FLOAT tensors.");
    }
    const int64_t n = x.element_count();
    std::vector<float> out(static_cast<size_t>(n));
    const float *src = x.AsFloat();
    for (int64_t i = 0; i < n; ++i) {
      out[static_cast<size_t>(i)] = std::fabs(src[static_cast<size_t>(i)]);
    }
    ++g_example_abs_run_count;
    rt::SetOutput(node, 0, rt::Tensor::FromFloat(node.output(0), x.shape, out, ctx.allocator()),
                  ctx);
  }
};

/// Installs :cpp:class:`ExampleAbsKernel` into onnx-light's shared
/// ``KernelDispatchTable`` for the CPU device. The empty domain is normalised
/// to the default ONNX domain, so this overrides the built-in ``Abs`` entry.
void RegisterExampleAbsKernel() {
  rt::RegisterKernelFn(
      /*domain=*/"", /*op_type=*/"Abs", sym::Device::kCPU,
      [](const NodeProto &node, rt::RuntimeContext &ctx) -> std::unique_ptr<rt::KernelBase> {
        auto kernel = std::make_unique<ExampleAbsKernel>(ctx.kernel_ctx());
        kernel->set_node(node);
        return kernel;
      });
}

// Builds a single-node graph ``y = Abs(x)`` with FLOAT input ``x`` and output
// ``y``.
GraphProto MakeAbsGraph() {
  GraphProto graph;
  graph.set_name("abs_graph");
  ValueInfoProto vi_x;
  vi_x.set_name("x");
  ValueInfoProto vi_y;
  vi_y.set_name("y");
  graph.ref_input().push_back(vi_x);
  graph.ref_output().push_back(vi_y);
  NodeProto node;
  node.set_op_type("Abs");
  node.add_input("x");
  node.add_output("y");
  graph.ref_node().push_back(node);
  return graph;
}

} // namespace example

int main() {
  using example::ExampleAbsKernel;

  // 1) Register onnx-light's built-in kernels so the rest of the dispatch
  //    table is populated, then install the override on top. This mirrors the
  //    onnx-light-cpu order (importing onnx-light registers the built-ins, then
  //    the extension registers its kernels). The override wins regardless of
  //    the order of these two calls: an explicit RegisterKernelFn replaces any
  //    existing entry, while the bulk built-in registration never clobbers a
  //    kernel that was already registered.
  ONNX_LIGHT_NAMESPACE::onnx_kernels::RegisterKernelFunctions();
  example::RegisterExampleAbsKernel();

  // Verify the kernel class exposes the expected "<library>:<device>:<domain>:<op_type>"
  // identifier before we rely on it below.
  if (std::string(ExampleAbsKernel::name) != "example:CPU:ai.onnx:Abs") {
    std::cerr << "ERROR: unexpected kernel class name '" << ExampleAbsKernel::name
              << "', expected 'example:CPU:ai.onnx:Abs'.\n";
    return 1;
  }

  std::cout << "Registered custom kernel class '" << ExampleAbsKernel::name
            << "' for op_type 'Abs' (default domain, CPU device).\n";

  // 2) Build a tiny model ``y = Abs(x)`` and run it through a RuntimeSession.
  const GraphProto graph = example::MakeAbsGraph();

  rt::RuntimeContext ctx(rt::KernelContext(rt::DefaultOpset(18)));
  ctx.Set("x", rt::Tensor::FromFloat("x", {4}, {-1.0f, 2.0f, -3.5f, 0.0f}));

  const rt::ExecutionPlan &plan = ctx.GetExecutionPlan(graph);
  rt::RuntimeSession session(plan);
  session.Run(ctx);

  // 3) Verify the output equals |x|.
  const rt::Tensor &y = ctx.Get("y");
  const std::vector<float> expected = {1.0f, 2.0f, 3.5f, 0.0f};
  if (y.element_count() != static_cast<int64_t>(expected.size())) {
    std::cerr << "ERROR: unexpected output element count " << y.element_count() << ".\n";
    return 1;
  }
  const float *py = y.AsFloat();
  for (size_t i = 0; i < expected.size(); ++i) {
    if (std::fabs(py[i] - expected[i]) > 1e-6f) {
      std::cerr << "ERROR: output[" << i << "] = " << py[i] << ", expected " << expected[i]
                << ".\n";
      return 1;
    }
  }

  // 4) Verify the custom kernel — not the built-in — actually ran.
  if (example::g_example_abs_run_count != 1) {
    std::cerr << "ERROR: expected ExampleAbsKernel to run exactly once, but it ran "
              << example::g_example_abs_run_count << " time(s). The built-in Abs kernel was "
              << "probably dispatched instead of the override.\n";
    return 1;
  }

  std::cout << "y = [";
  for (size_t i = 0; i < expected.size(); ++i) {
    std::cout << (i ? ", " : "") << py[i];
  }
  std::cout << "]\n";
  std::cout << "PASS: the custom '" << ExampleAbsKernel::name
            << "' kernel ran and produced the expected output.\n";
  return 0;
}
