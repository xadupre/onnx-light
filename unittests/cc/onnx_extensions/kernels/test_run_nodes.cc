// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/test_case.h"
#include "onnx_core/compute/compute_context.h"
#include "onnx_core/compute/execute_action.h"
#include "onnx_core/compute/inplace_reuse.h"
#include "onnx_core/compute/peak_memory.h"
#include "onnx_core/compute/raw_buffer_allocator.h"
#include "onnx_core/runtime/kernel_context.h"
#include "onnx_core/runtime/kernel_dispatch_table.h"
#include "onnx_core/runtime/kernel_tuning.h"
#include "onnx_core/runtime/node_helpers.h"
#include "onnx_core/runtime/run_nodes.h"
#include "onnx_core/runtime/runtime_session.h"
#include "onnx_core/runtime/simple_sequence.h"
#include "onnx_core/runtime/simple_tensor.h"
#include "onnx_extensions/kernels/kernel_dispatch_table.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"
#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"
#include "onnx_extensions/kernels/kernels/rt/include_rt_kernels.h"
#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_extensions/kernels/kernels/training/include_training_kernels.h"
#include "onnx_proto/onnx.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using core::backend_test::DefaultOpset;
using core::runtime::DataType;
using core::runtime::ExecutionPlan;
using core::runtime::KernelDispatchTable;
using core::runtime::RegisterModelFunctions;
using core::runtime::RunModel;
using core::runtime::RunNode;
using core::runtime::RuntimeContext;
using core::runtime::RuntimeParameters;
using core::runtime::RuntimeSession;
using core::runtime::RuntimeSessionOptions;
using core::runtime::Sequence;
using core::runtime::SliceTensorAlongAxis;
using core::runtime::Tensor;
using core::runtime::TensorFromProto;
using core::runtime::TensorMap;
using core::runtime::Tensors;
using onnx_kernels::kernel::KernelContext;

namespace {

class EnvVarGuard {
public:
  explicit EnvVarGuard(const char *name) : name_(name) {
    const char *value = std::getenv(name_.c_str());
    if (value != nullptr) {
      had_value_ = true;
      value_ = value;
    }
  }

  ~EnvVarGuard() {
    if (had_value_) {
#ifdef _WIN32
      _putenv_s(name_.c_str(), value_.c_str());
#else
      setenv(name_.c_str(), value_.c_str(), 1);
#endif
      return;
    }
#ifdef _WIN32
    _putenv_s(name_.c_str(), "");
#else
    unsetenv(name_.c_str());
#endif
  }

  EnvVarGuard(const EnvVarGuard &) = delete;
  EnvVarGuard &operator=(const EnvVarGuard &) = delete;
  EnvVarGuard(EnvVarGuard &&) = delete;
  EnvVarGuard &operator=(EnvVarGuard &&) = delete;

private:
  std::string name_;
  bool had_value_ = false;
  std::string value_;
};

class TempFileCleanupGuard {
public:
  explicit TempFileCleanupGuard(std::filesystem::path path) : path_(std::move(path)) {}

  ~TempFileCleanupGuard() {
    std::error_code ec;
    std::filesystem::remove(path_, ec);
  }

  TempFileCleanupGuard(const TempFileCleanupGuard &) = delete;
  TempFileCleanupGuard &operator=(const TempFileCleanupGuard &) = delete;
  TempFileCleanupGuard(TempFileCleanupGuard &&) = delete;
  TempFileCleanupGuard &operator=(TempFileCleanupGuard &&) = delete;

private:
  std::filesystem::path path_;
};

// Minimal :cpp:class:`core::runtime::KernelBase` used by the synthetic-op tests
// below: it carries a per-run closure so each test can register a
// ``NodeKernelFn`` factory (construction) that returns a kernel whose
// :cpp:func:`Run` (execution) is tracked separately.
class TestLambdaKernel : public core::runtime::KernelBase {
public:
  using Fn = std::function<void(const NodeProto &node, RuntimeContext &rt)>;
  TestLambdaKernel(const NodeProto &node, Fn fn)
      : core::runtime::KernelBase(core::runtime::KernelContext{}), fn_(std::move(fn)) {
    set_node(node);
  }
  void Run(RuntimeContext &rt) override { fn_(*node_, rt); }

private:
  Fn fn_;
};

class TestTunableKernel : public core::runtime::KernelBase {
public:
  TestTunableKernel(const NodeProto &node, core::runtime::KernelTuningKey key)
      : core::runtime::KernelBase(core::runtime::KernelContext{}), key_(std::move(key)) {
    set_node(node);
  }

  core::runtime::KernelTuningKey TuningKey(int32_t element_type) const override {
    core::runtime::KernelTuningKey key = key_;
    key.element_type = element_type;
    return key;
  }

  void Configure(const core::runtime::KernelTuningParameters &parameters) override {
    configured_value_ = parameters.Get<int64_t>("algorithm.threshold");
  }

  void Run(RuntimeContext &rt) override {
    rt.Set(node_->output(0),
           Tensor::FromInt64(node_->output(0), {1}, std::vector<int64_t>{configured_value_}));
  }

private:
  core::runtime::KernelTuningKey key_;
  int64_t configured_value_ = -1;
};

} // namespace

static_assert(std::string_view(core::runtime::RuntimeEventActionName(
                  core::runtime::RuntimeEventAction::kRunNode)) == "run_node");
static_assert(std::string_view(core::runtime::RuntimeEventKindName(
                  core::runtime::RuntimeEventKind::kOutput)) == "output");

namespace Test {

namespace {

class CountingAllocator : public core::runtime::RawBufferAllocator {
public:
  explicit CountingAllocator(size_t capacity) : pool_(capacity) {}

  core::runtime::RawBuffer *Allocate(size_t n_bytes) override {
    ++allocate_calls_;
    return pool_.Allocate(n_bytes);
  }

  void Free(core::runtime::RawBuffer *buf) override {
    ++free_calls_;
    pool_.Free(buf);
  }

  size_t TotalAllocatedSize() const override { return pool_.TotalAllocatedSize(); }

  size_t PeakAllocatedSize() const override { return pool_.PeakAllocatedSize(); }

  void ResetPeak() override { pool_.ResetPeak(); }

  size_t allocated_count() const noexcept { return pool_.allocated_count(); }
  size_t allocate_calls() const noexcept { return allocate_calls_; }
  size_t free_calls() const noexcept { return free_calls_; }

private:
  core::runtime::SimpleRawBufferAllocator pool_;
  size_t allocate_calls_ = 0;
  size_t free_calls_ = 0;
};

// Seeds `graph`'s initializers into `rt` (without overriding any name the
// caller already provided) and then runs `graph.node()` by building the
// graph's ExecutionPlan and driving it through a RuntimeSession. This
// mirrors the production `RunGraphNodesViaSession` helper used internally
// by `SubgraphSession` and the control-flow kernels, giving these
// tests the same "build a plan, then run it through a session" flow every
// call site now uses.
void RunGraphViaSession(const GraphProto &graph, RuntimeContext &rt) {
  const auto &inits = graph.initializer();
  for (size_t i = 0; i < inits.size(); ++i) {
    const TensorProto &tp = inits[i];
    const std::string init_name = tp.name();
    if (!rt.Has(init_name)) {
      rt.Set(init_name, TensorFromProto(tp), core::runtime::RuntimeEventKind::kInitializer);
    }
  }
  const ExecutionPlan &plan = rt.GetExecutionPlan(graph);
  RuntimeSession session(plan);
  session.Run(rt);
}

// Registers `model`'s local functions in `rt` and then runs `model.graph()`
// the same way `RunGraphViaSession` runs a bare graph. This mirrors the
// production `RunModel` replacement: callers register model-local functions
// via `RegisterModelFunctions` and then drive the model's graph through
// their own ExecutionPlan/RuntimeSession.
void RunModelViaSession(const ModelProto &model, RuntimeContext &rt) {
  RegisterModelFunctions(model, rt);
  RunGraphViaSession(model.ref_graph(), rt);
}

// Builds a single-node ``NodeProto`` of type ``op_type`` with the
// requested input and output names.
NodeProto MakeNode(const std::string &op_type, const std::vector<std::string> &inputs,
                   const std::vector<std::string> &outputs, const std::string &domain = "") {
  NodeProto node;
  node.set_op_type(op_type);
  if (!domain.empty()) {
    node.set_domain(domain);
  }
  for (const auto &name : inputs) {
    node.add_input(name);
  }
  for (const auto &name : outputs) {
    node.add_output(name);
  }
  return node;
}

// Declares a sequence-typed graph input/output named ``name`` carrying
// ``FLOAT`` tensor elements. Used to build the body subgraph of a Loop
// node whose loop-carried state is sequence-typed.
void AddSequenceFloatValueInfo(ValueInfoProto *vi, const std::string &name) {
  vi->set_name(name);
  TypeProto *tp = vi->ref_type().add_sequence_type()->add_elem_type();
  tp->add_tensor_type()->set_elem_type(static_cast<int>(TensorProto::DataType::FLOAT));
}

// Appends a ``Constant`` node producing a 1-D INT64 tensor ``[0]`` under
// the output name ``out`` (used as the ``axes`` input of ``Unsqueeze``).
void AddInt64AxesConstant(GraphProto &g, const std::string &out) {
  NodeProto *n = g.add_node();
  n->set_op_type("Constant");
  n->add_output(out);
  AttributeProto *a = n->add_attribute();
  a->set_name("value");
  a->set_type(AttributeProto::AttributeType::TENSOR);
  TensorProto *t = a->add_t();
  t->set_data_type(TensorProto::DataType::INT64);
  t->add_dims(1);
  t->add_int64_data(0);
}

// Appends nodes that turn the INT64 scalar ``iter`` into a ``FLOAT[1]``
// tensor under the output name ``out`` (Unsqueeze to ``[1]`` then Cast).
void AddIterAsFloat1D(GraphProto &g, const std::string &out) {
  AddInt64AxesConstant(g, "axes");
  {
    NodeProto *n = g.add_node();
    n->set_op_type("Unsqueeze");
    n->add_input("iter");
    n->add_input("axes");
    n->add_output("iter_1d");
  }
  {
    NodeProto *n = g.add_node();
    n->set_op_type("Cast");
    n->add_input("iter_1d");
    n->add_output(out);
    AttributeProto *a = n->add_attribute();
    a->set_name("to");
    a->set_type(AttributeProto::AttributeType::INT);
    a->set_i(static_cast<int64_t>(TensorProto::DataType::FLOAT));
  }
}

// Body subgraph for a Loop whose only loop-carried state is a
// sequence. Inputs are ``(iter, cond_in, seq_in)``; outputs are
// ``(cond_out, seq_out)`` where ``seq_out = SequenceInsert(seq_in,
// (float)iter)`` so the sequence grows by one ``FLOAT[1]`` element per
// iteration.
GraphProto BuildSequenceLoopBody() {
  GraphProto body;
  body.set_name("seq_loop_body");
  body.add_input()->set_name("iter");
  body.add_input()->set_name("cond_in");
  AddSequenceFloatValueInfo(body.add_input(), "seq_in");

  {
    NodeProto *n = body.add_node();
    n->set_op_type("Identity");
    n->add_input("cond_in");
    n->add_output("cond_out");
  }
  AddIterAsFloat1D(body, "val");
  {
    NodeProto *n = body.add_node();
    n->set_op_type("SequenceInsert");
    n->add_input("seq_in");
    n->add_input("val");
    n->add_output("seq_out");
  }

  body.add_output()->set_name("cond_out");
  AddSequenceFloatValueInfo(body.add_output(), "seq_out");
  return body;
}

// Body subgraph for a Loop with mixed state: a tensor loop-carried
// accumulator and a sequence loop-carried state, plus one scan output.
// Inputs are ``(iter, cond_in, acc_in, seq_in)``; outputs are
// ``(cond_out, acc_out, seq_out, scan_out)`` where
// ``acc_out = acc_in + 1``, ``scan_out = acc_out`` and
// ``seq_out = SequenceInsert(seq_in, (float)iter)``.
GraphProto BuildMixedSequenceLoopBody() {
  GraphProto body;
  body.set_name("mixed_seq_loop_body");
  body.add_input()->set_name("iter");
  body.add_input()->set_name("cond_in");
  body.add_input()->set_name("acc_in");
  AddSequenceFloatValueInfo(body.add_input(), "seq_in");

  {
    NodeProto *n = body.add_node();
    n->set_op_type("Identity");
    n->add_input("cond_in");
    n->add_output("cond_out");
  }
  {
    NodeProto *n = body.add_node();
    n->set_op_type("Constant");
    n->add_output("one");
    AttributeProto *a = n->add_attribute();
    a->set_name("value");
    a->set_type(AttributeProto::AttributeType::TENSOR);
    TensorProto *t = a->add_t();
    t->set_data_type(TensorProto::DataType::FLOAT);
    t->add_float_data(1.0f);
  }
  {
    NodeProto *n = body.add_node();
    n->set_op_type("Add");
    n->add_input("acc_in");
    n->add_input("one");
    n->add_output("acc_out");
  }
  {
    NodeProto *n = body.add_node();
    n->set_op_type("Identity");
    n->add_input("acc_out");
    n->add_output("scan_out");
  }
  AddIterAsFloat1D(body, "val");
  {
    NodeProto *n = body.add_node();
    n->set_op_type("SequenceInsert");
    n->add_input("seq_in");
    n->add_input("val");
    n->add_output("seq_out");
  }

  body.add_output()->set_name("cond_out");
  body.add_output()->set_name("acc_out");
  AddSequenceFloatValueInfo(body.add_output(), "seq_out");
  body.add_output()->set_name("scan_out");
  return body;
}

// Builds a ``Loop`` node binding ``inputs``/``outputs`` and attaching
// ``body`` as its ``body`` graph attribute.
NodeProto MakeLoopNode(const std::vector<std::string> &inputs,
                       const std::vector<std::string> &outputs, GraphProto body) {
  NodeProto loop = MakeNode("Loop", inputs, outputs);
  AttributeProto *body_attr = loop.add_attribute();
  body_attr->set_name("body");
  body_attr->set_type(AttributeProto::AttributeType::GRAPH);
  *body_attr->add_g() = std::move(body);
  return loop;
}

} // namespace

TEST(RunNodes, DispatchTableContainsRegisteredOps) {
  const auto &table = KernelDispatchTable();
  // Spot-check the initial registered baseline of element-wise math ops.
  EXPECT_NE(table.find("ai.onnx:Add"), table.end());
  EXPECT_NE(table.find("ai.onnx:Sub"), table.end());
  EXPECT_NE(table.find("ai.onnx:Mul"), table.end());
  EXPECT_NE(table.find("ai.onnx:Div"), table.end());
  EXPECT_NE(table.find("ai.onnx:Neg"), table.end());
  EXPECT_NE(table.find("ai.onnx:Abs"), table.end());

  // Spot-check the extended registration set covering the rest of the
  // unary / binary / variadic / attribute-driven math + logical
  // kernels (see ``onnx_kernels/kernel_dispatch_table.cc``).
  // Unary math, no attributes.
  EXPECT_NE(table.find("ai.onnx:Cos"), table.end());
  EXPECT_NE(table.find("ai.onnx:Erf"), table.end());
  EXPECT_NE(table.find("ai.onnx:Sigmoid"), table.end());
  EXPECT_NE(table.find("ai.onnx:Tanh"), table.end());
  // Binary math (Gemm is attribute-driven; MatMul/Pow are not).
  EXPECT_NE(table.find("ai.onnx:Gemm"), table.end());
  EXPECT_NE(table.find("ai.onnx:MatMul"), table.end());
  EXPECT_NE(table.find("ai.onnx:Pow"), table.end());
  // Variadic reducers.
  EXPECT_NE(table.find("ai.onnx:Sum"), table.end());
  EXPECT_NE(table.find("ai.onnx:Max"), table.end());
  EXPECT_NE(table.find("ai.onnx:Min"), table.end());
  EXPECT_NE(table.find("ai.onnx:Mean"), table.end());
  // Reduction operators.
  EXPECT_NE(table.find("ai.onnx:ArgMax"), table.end());
  EXPECT_NE(table.find("ai.onnx:ArgMin"), table.end());
  EXPECT_NE(table.find("ai.onnx:ReduceL1"), table.end());
  EXPECT_NE(table.find("ai.onnx:ReduceL2"), table.end());
  EXPECT_NE(table.find("ai.onnx:ReduceLogSum"), table.end());
  EXPECT_NE(table.find("ai.onnx:ReduceLogSumExp"), table.end());
  EXPECT_NE(table.find("ai.onnx:ReduceMax"), table.end());
  EXPECT_NE(table.find("ai.onnx:ReduceMean"), table.end());
  EXPECT_NE(table.find("ai.onnx:ReduceMin"), table.end());
  EXPECT_NE(table.find("ai.onnx:ReduceProd"), table.end());
  EXPECT_NE(table.find("ai.onnx:ReduceSum"), table.end());
  EXPECT_NE(table.find("ai.onnx:ReduceSumSquare"), table.end());
  // Attribute-driven kernels.
  EXPECT_NE(table.find("ai.onnx:Softmax"), table.end());
  EXPECT_NE(table.find("ai.onnx:LeakyRelu"), table.end());
  EXPECT_NE(table.find("ai.onnx:HardSigmoid"), table.end());
  EXPECT_NE(table.find("ai.onnx:Selu"), table.end());
  EXPECT_NE(table.find("ai.onnx:Gelu"), table.end());
  EXPECT_NE(table.find("ai.onnx:Mod"), table.end());
  EXPECT_NE(table.find("ai.onnx:Clip"), table.end());
  EXPECT_NE(table.find("ai.onnx:Attention"), table.end());
  EXPECT_NE(table.find("ai.onnx:GRU"), table.end());
  EXPECT_NE(table.find("ai.onnx:NonMaxSuppression"), table.end());
  EXPECT_NE(table.find("ai.onnx:NonZero"), table.end());
  EXPECT_NE(table.find("ai.onnx:IsInf"), table.end());
  EXPECT_NE(table.find("ai.onnx:BitShift"), table.end());
  EXPECT_NE(table.find("ai.onnx:BitCast"), table.end());
  EXPECT_NE(table.find("ai.onnx:Einsum"), table.end());
  EXPECT_NE(table.find("ai.onnx:DFT"), table.end());
  EXPECT_NE(table.find("ai.onnx:TopK"), table.end());
  // Quantization kernels.
  EXPECT_NE(table.find("ai.onnx:QuantizeLinear"), table.end());
  EXPECT_NE(table.find("ai.onnx:DequantizeLinear"), table.end());
  EXPECT_NE(table.find("ai.onnx:DynamicQuantizeLinear"), table.end());
  // Generator kernels.
  EXPECT_NE(table.find("ai.onnx:EyeLike"), table.end());
  EXPECT_NE(table.find("ai.onnx:AffineGrid"), table.end());
  EXPECT_NE(table.find("ai.onnx:ImageDecoder"), table.end());
  // Tensor shape kernels.
  EXPECT_NE(table.find("ai.onnx:Cast"), table.end());
  EXPECT_NE(table.find("ai.onnx:Shape"), table.end());
  EXPECT_NE(table.find("ai.onnx:Size"), table.end());
  EXPECT_NE(table.find("ai.onnx:DepthToSpace"), table.end());
  EXPECT_NE(table.find("ai.onnx:Gather"), table.end());
  EXPECT_NE(table.find("ai.onnx:GatherND"), table.end());
  EXPECT_NE(table.find("ai.onnx:Pad"), table.end());
  // Logical / bitwise kernels.
  EXPECT_NE(table.find("ai.onnx:And"), table.end());
  EXPECT_NE(table.find("ai.onnx:Or"), table.end());
  EXPECT_NE(table.find("ai.onnx:Xor"), table.end());
  EXPECT_NE(table.find("ai.onnx:Not"), table.end());
  EXPECT_NE(table.find("ai.onnx:Equal"), table.end());
  EXPECT_NE(table.find("ai.onnx:Greater"), table.end());
  EXPECT_NE(table.find("ai.onnx:Less"), table.end());
  EXPECT_NE(table.find("ai.onnx:Where"), table.end());
  EXPECT_NE(table.find("ai.onnx:IsNaN"), table.end());
  EXPECT_NE(table.find("ai.onnx:BitwiseAnd"), table.end());
  EXPECT_NE(table.find("ai.onnx:BitwiseNot"), table.end());
  // ai.onnx.preview.training optimizers.
  EXPECT_NE(table.find("ai.onnx.preview.training:Adagrad"), table.end());
  EXPECT_NE(table.find("ai.onnx.preview.training:Adam"), table.end());
  EXPECT_NE(table.find("ai.onnx.preview.training:Momentum"), table.end());
  // ai.onnx.preview kernels.
  EXPECT_NE(table.find("ai.onnx.preview:FlexAttention"), table.end());
  // ai.onnx.ml kernels.
  EXPECT_NE(table.find("ai.onnx.ml:SVMRegressor"), table.end());
  EXPECT_NE(table.find("ai.onnx.ml:SVMClassifier"), table.end());
  EXPECT_NE(table.find("ai.onnx.ml:ArrayFeatureExtractor"), table.end());
  EXPECT_NE(table.find("ai.onnx.ml:Binarizer"), table.end());
  EXPECT_NE(table.find("ai.onnx.ml:Imputer"), table.end());
  EXPECT_NE(table.find("ai.onnx.ml:FeatureVectorizer"), table.end());
  EXPECT_NE(table.find("ai.onnx.ml:LabelEncoder"), table.end());
  // Linear attention (opset 27).
  EXPECT_NE(table.find("ai.onnx:LinearAttention"), table.end());
  // Normalization kernels.
  EXPECT_NE(table.find("ai.onnx:BatchNormalization"), table.end());
  // Sequence operators (opset 11+).
  EXPECT_NE(table.find("ai.onnx:SequenceConstruct"), table.end());
  EXPECT_NE(table.find("ai.onnx:SequenceEmpty"), table.end());
  EXPECT_NE(table.find("ai.onnx:SequenceAt"), table.end());
  EXPECT_NE(table.find("ai.onnx:SequenceErase"), table.end());
  EXPECT_NE(table.find("ai.onnx:SequenceInsert"), table.end());
  EXPECT_NE(table.find("ai.onnx:SequenceLength"), table.end());
  EXPECT_NE(table.find("ai.onnx:ConcatFromSequence"), table.end());
  EXPECT_NE(table.find("ai.onnx:SplitToSequence"), table.end());
  // Optional kernels (opset 15+).
  EXPECT_NE(table.find("ai.onnx:Optional"), table.end());
  EXPECT_NE(table.find("ai.onnx:OptionalGetElement"), table.end());
  EXPECT_NE(table.find("ai.onnx:OptionalHasElement"), table.end());
  // Text kernels (ai.onnx).
  EXPECT_NE(table.find("ai.onnx:RegexFullMatch"), table.end());
}

TEST(RunNodes, RunNodeSingleAdd) {
  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.tensors()["x"] = Tensor::FromFloat("x", {3}, {1.0f, 2.0f, 3.0f});
  rt.tensors()["y"] = Tensor::FromFloat("y", {3}, {10.0f, 20.0f, 30.0f});
  NodeProto node = MakeNode("Add", {"x", "y"}, {"z"});
  RunNode(node, rt);
  ASSERT_NE(rt.tensors().find("z"), rt.tensors().end());
  const Tensor &z = rt.tensors()["z"];
  EXPECT_EQ(z.name, "z");
  EXPECT_EQ(z.shape, std::vector<int64_t>({3}));
  ASSERT_EQ(z.element_count(), 3);
  const float *got = z.AsFloat();
  EXPECT_FLOAT_EQ(got[0], 11.0f);
  EXPECT_FLOAT_EQ(got[1], 22.0f);
  EXPECT_FLOAT_EQ(got[2], 33.0f);
}

TEST(RunNodes, RunNodeNonCpuDeviceWithoutKernelThrows) {
  // The C++ ReferenceEvaluator only ships CPU kernels. Selecting a non-CPU
  // device makes the device-qualified dispatch key miss so RunNode fails with
  // a diagnostic naming the device instead of silently running on the CPU
  // kernel.
  const core::symbolic::Device gpu = core::symbolic::MakeGPUDevice(0);
  RuntimeContext rt(KernelContext(DefaultOpset(18)),
                    core::runtime::RuntimeContextOptions{.device = gpu});
  EXPECT_EQ(rt.device(), gpu);
  rt.tensors()["x"] = Tensor::FromFloat("x", {3}, {1.0f, 2.0f, 3.0f});
  rt.tensors()["y"] = Tensor::FromFloat("y", {3}, {10.0f, 20.0f, 30.0f});
  NodeProto node = MakeNode("Add", {"x", "y"}, {"z"});
  try {
    RunNode(node, rt);
    FAIL() << "expected RunNode to throw for a non-CPU device";
  } catch (const std::invalid_argument &e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("unsupported op_type 'Add'"), std::string::npos) << msg;
    EXPECT_NE(msg.find("on device 'GPU0'"), std::string::npos) << msg;
  }
}

TEST(RunNodes, RunNodeDispatchesDeviceQualifiedKernel) {
  // A kernel registered for a specific device is keyed separately from the
  // CPU/default entry (see :cpp:func:`RegisterKernelFn`). Running a context
  // pinned to that device must resolve the device-qualified kernel rather than
  // falling back to the CPU one.
  using core::runtime::NodeKernelFn;
  using core::runtime::RegisterKernelFn;

  const core::symbolic::Device gpu = core::symbolic::MakeGPUDevice(0);
  const std::string domain = "test.onnxlight.device_dispatch";
  bool cpu_invoked = false;
  bool gpu_invoked = false;
  RegisterKernelFn(domain, "DeviceOp", core::symbolic::Device::kCPU,
                   [&cpu_invoked](const NodeProto &node,
                                  RuntimeContext &) -> std::unique_ptr<core::runtime::KernelBase> {
                     return std::make_unique<TestLambdaKernel>(
                         node, [&cpu_invoked](const NodeProto &node, RuntimeContext &rt) {
                           cpu_invoked = true;
                           rt.Set(node.output(0), rt.Get(node.input(0)));
                         });
                   });
  RegisterKernelFn(domain, "DeviceOp", gpu,
                   [&gpu_invoked](const NodeProto &node,
                                  RuntimeContext &) -> std::unique_ptr<core::runtime::KernelBase> {
                     return std::make_unique<TestLambdaKernel>(
                         node, [&gpu_invoked](const NodeProto &node, RuntimeContext &rt) {
                           gpu_invoked = true;
                           rt.Set(node.output(0), rt.Get(node.input(0)));
                         });
                   });

  RuntimeContext rt(KernelContext(DefaultOpset(18)),
                    core::runtime::RuntimeContextOptions{.device = gpu});
  rt.tensors()["x"] = Tensor::FromFloat("x", {2}, {1.0f, 2.0f});
  NodeProto node = MakeNode("DeviceOp", {"x"}, {"y"}, domain);
  RunNode(node, rt);
  EXPECT_TRUE(gpu_invoked);
  EXPECT_FALSE(cpu_invoked);
  ASSERT_NE(rt.tensors().find("y"), rt.tensors().end());
}

TEST(RunNodes, RunNodeClearResetsStateButKeepsSettings) {
  RuntimeContext rt(KernelContext(DefaultOpset(18)),
                    core::runtime::RuntimeContextOptions{.events_enabled = true});
  rt.set_release_intermediates(true);
  rt.tensors()["x"] = Tensor::FromFloat("x", {3}, {1.0f, 2.0f, 3.0f});
  rt.tensors()["y"] = Tensor::FromFloat("y", {3}, {10.0f, 20.0f, 30.0f});
  NodeProto node = MakeNode("Add", {"x", "y"}, {"z"});
  RunNode(node, rt);
  ASSERT_FALSE(rt.tensors().empty());
  ASSERT_FALSE(rt.events().empty());

  rt.Clear();

  // Tensor map, sequence map and event log are reset.
  EXPECT_TRUE(rt.tensors().empty());
  EXPECT_TRUE(rt.sequences().empty());
  EXPECT_TRUE(rt.events().empty());
  EXPECT_EQ(rt.current_node_index(), -1);
  // Settings survive the reset.
  EXPECT_TRUE(rt.events_enabled());
  EXPECT_TRUE(rt.release_intermediates());

  // The context can be reused for a fresh run after clearing.
  rt.tensors()["x"] = Tensor::FromFloat("x", {3}, {4.0f, 5.0f, 6.0f});
  rt.tensors()["y"] = Tensor::FromFloat("y", {3}, {40.0f, 50.0f, 60.0f});
  RunNode(node, rt);
  ASSERT_NE(rt.tensors().find("z"), rt.tensors().end());
  const float *got = rt.tensors()["z"].AsFloat();
  ASSERT_EQ(rt.tensors()["z"].element_count(), 3);
  EXPECT_FLOAT_EQ(got[0], 44.0f);
  EXPECT_FLOAT_EQ(got[1], 55.0f);
  EXPECT_FLOAT_EQ(got[2], 66.0f);
}

TEST(RunNodes, VerboseRunNodeLogsThroughLoggerDestination) {
  const uint64_t time_seed =
      static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
  const uint64_t thread_seed =
      static_cast<uint64_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
  std::mt19937_64 rng(time_seed ^ thread_seed);
  const std::filesystem::path log_path =
      std::filesystem::temp_directory_path() /
      ("onnx_light_verbose_progress_" + std::to_string(rng()) + ".log");
  TempFileCleanupGuard cleanup(log_path);
  EnvVarGuard guard("ONNX_LIGHT_LOG");
#ifdef _WIN32
  _putenv_s("ONNX_LIGHT_LOG", log_path.string().c_str());
#else
  setenv("ONNX_LIGHT_LOG", log_path.string().c_str(), 1);
#endif

  RuntimeContext rt(KernelContext(DefaultOpset(18)),
                    core::runtime::RuntimeContextOptions{.verbose = 1});

  rt.RegisterCustomKernel(
      "com.acme", "VerboseProbe",
      [](const NodeProto &node,
         RuntimeContext & /*unused*/) -> std::unique_ptr<core::runtime::KernelBase> {
        // Intentional no-op kernel: this test only verifies verbose logging
        // plumbing in RunNode.
        return std::make_unique<TestLambdaKernel>(
            node, [](const NodeProto & /*unused*/, RuntimeContext & /*unused*/) {});
      });

  NodeProto node = MakeNode("VerboseProbe", {}, {}, "com.acme");
  RunNode(node, rt);

  ASSERT_TRUE(std::filesystem::exists(log_path));
  std::ifstream in(log_path);
  ASSERT_TRUE(in.is_open());
  std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  EXPECT_NE(content.find("[ReferenceEvaluator]"), std::string::npos);
  EXPECT_NE(content.find("com.acme::VerboseProbe() -> ()"), std::string::npos);
}

TEST(RunNodes, RunNodeGemmWithoutBiasUsesSchemaDefaults) {
  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.tensors()["a"] = Tensor::FromFloat("a", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  rt.tensors()["b"] = Tensor::FromFloat("b", {2, 2}, {5.0f, 6.0f, 7.0f, 8.0f});
  NodeProto node = MakeNode("Gemm", {"a", "b"}, {"y"});
  RunNode(node, rt);
  ASSERT_NE(rt.tensors().find("y"), rt.tensors().end());
  const Tensor &y = rt.tensors()["y"];
  EXPECT_EQ(y.shape, std::vector<int64_t>({2, 2}));
  ASSERT_EQ(y.element_count(), 4);
  const float *got = y.AsFloat();
  EXPECT_FLOAT_EQ(got[0], 19.0f);
  EXPECT_FLOAT_EQ(got[1], 22.0f);
  EXPECT_FLOAT_EQ(got[2], 43.0f);
  EXPECT_FLOAT_EQ(got[3], 50.0f);
}

TEST(RunNodes, RunNodeNormalisesDefaultDomain) {
  // The default ONNX domain is the empty string. The dispatcher must
  // normalise it to ``ai.onnx`` before looking up the kernel.
  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.tensors()["x"] = Tensor::FromFloat("x", {2}, {-1.5f, 2.5f});
  NodeProto node = MakeNode("Abs", {"x"}, {"y"}); // empty domain
  EXPECT_TRUE(node.domain().empty());
  RunNode(node, rt);
  const float *got = rt.tensors()["y"].AsFloat();
  ASSERT_EQ(rt.tensors()["y"].element_count(), 2);
  EXPECT_FLOAT_EQ(got[0], 1.5f);
  EXPECT_FLOAT_EQ(got[1], 2.5f);
}

TEST(RunNodes, RunNodeCastFromDispatchTable) {
  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.tensors()["x"] = Tensor::FromFloat("x", {3}, {-1.5f, 0.0f, 2.75f});
  NodeProto node = MakeNode("Cast", {"x"}, {"y"});
  AttributeProto *to = node.add_attribute();
  to->set_name("to");
  to->set_type(AttributeProto::INT);
  to->set_i(static_cast<int64_t>(DataType::INT32));

  RunNode(node, rt);

  const Tensor &y = rt.tensors().at("y");
  EXPECT_EQ(y.data_type, static_cast<int32_t>(DataType::INT32));
  EXPECT_EQ(y.shape, std::vector<int64_t>({3}));
  const int32_t *got = y.AsInt32();
  EXPECT_EQ(got[0], -1);
  EXPECT_EQ(got[1], 0);
  EXPECT_EQ(got[2], 2);
}

TEST(RunNodes, RunNodeBitCastFromDispatchTable) {
  RuntimeContext rt(KernelContext(DefaultOpset(26)));
  rt.tensors()["x"] = Tensor::FromFloat("x", {3}, {0.0f, 1.0f, -1.0f});
  NodeProto node = MakeNode("BitCast", {"x"}, {"y"});
  AttributeProto *to = node.add_attribute();
  to->set_name("to");
  to->set_type(AttributeProto::INT);
  to->set_i(static_cast<int64_t>(DataType::INT32));

  RunNode(node, rt);

  const Tensor &y = rt.tensors().at("y");
  EXPECT_EQ(y.data_type, static_cast<int32_t>(DataType::INT32));
  EXPECT_EQ(y.shape, std::vector<int64_t>({3}));
  const int32_t *got = y.AsInt32();
  EXPECT_EQ(got[0], 0);
  EXPECT_EQ(got[1], 1065353216);
  EXPECT_EQ(got[2], -1082130432);
}

TEST(RunNodes, RunNodeNonMaxSuppressionFromDispatchTable) {
  // NonMaxSuppression was introduced in ONNX opset 10; use opset 11 to match
  // the other NonMaxSuppression kernel tests in this repository.
  RuntimeContext rt(KernelContext(DefaultOpset(11)));
  rt.tensors()["boxes"] =
      Tensor::FromFloat("boxes", {1, 2, 4}, {0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 10.0f, 1.0f, 11.0f});
  rt.tensors()["scores"] = Tensor::FromFloat("scores", {1, 1, 2}, {0.9f, 0.8f});
  rt.tensors()["max_output_boxes_per_class"] =
      Tensor::FromInt64("max_output_boxes_per_class", {1}, {10});
  rt.tensors()["iou_threshold"] = Tensor::FromFloat("iou_threshold", {1}, {0.5f});
  rt.tensors()["score_threshold"] = Tensor::FromFloat("score_threshold", {1}, {0.0f});

  NodeProto node = MakeNode(
      "NonMaxSuppression",
      {"boxes", "scores", "max_output_boxes_per_class", "iou_threshold", "score_threshold"},
      {"selected_indices"});
  RunNode(node, rt);

  const Tensor &selected = rt.tensors().at("selected_indices");
  EXPECT_EQ(selected.shape, (std::vector<int64_t>{2, 3}));
  const int64_t *py = selected.AsInt64();
  const std::vector<int64_t> expected = {0, 0, 0, 0, 0, 1};
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(py[i], expected[i]);
  }
}

TEST(RunNodes, RunNodeNonZeroFromDispatchTable) {
  RuntimeContext rt(KernelContext(DefaultOpset(13)));
  rt.tensors()["x"] = Tensor::FromFloat("x", {2, 3}, {1.0f, 0.0f, 2.0f, 0.0f, 3.0f, 0.0f});

  NodeProto node = MakeNode("NonZero", {"x"}, {"y"});
  RunNode(node, rt);

  const Tensor &y = rt.tensors().at("y");
  EXPECT_EQ(y.shape, (std::vector<int64_t>{2, 3}));
  const int64_t *py = y.AsInt64();
  const std::vector<int64_t> expected = {0, 0, 1, 0, 2, 1};
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(py[i], expected[i]);
  }
}

TEST(RunNodes, RunNodeQuantizeLinearFromDispatchTable) {
  RuntimeContext rt(KernelContext(DefaultOpset(13)));
  rt.tensors()["x"] = Tensor::FromFloat("x", {4}, {0.0f, 2.0f, 3.0f, 254.0f});
  rt.tensors()["y_scale"] = Tensor::FromFloat("y_scale", {}, {2.0f});

  NodeProto node = MakeNode("QuantizeLinear", {"x", "y_scale"}, {"y"});
  RunNode(node, rt);

  const Tensor &y = rt.tensors().at("y");
  EXPECT_EQ(y.data_type, static_cast<int32_t>(core::runtime::DataType::UINT8));
  EXPECT_EQ(y.shape, std::vector<int64_t>({4}));
  ASSERT_EQ(y.element_count(), 4);
  const std::uint8_t *py = reinterpret_cast<const std::uint8_t *>(y.bytes());
  EXPECT_EQ(py[0], 0);
  EXPECT_EQ(py[1], 1);
  EXPECT_EQ(py[2], 2);
  EXPECT_EQ(py[3], 127);
}

TEST(RunNodes, RunNodeDequantizeLinearFromDispatchTable) {
  RuntimeContext rt(KernelContext(DefaultOpset(13)));
  rt.tensors()["x"] = Tensor::FromUint8("x", {3}, {0, 3, 128});
  rt.tensors()["x_scale"] = Tensor::FromFloat("x_scale", {}, {2.0f});
  rt.tensors()["x_zero_point"] = Tensor::FromUint8("x_zero_point", {}, {128});

  NodeProto node = MakeNode("DequantizeLinear", {"x", "x_scale", "x_zero_point"}, {"y"});
  RunNode(node, rt);

  const Tensor &y = rt.tensors().at("y");
  EXPECT_EQ(y.shape, std::vector<int64_t>({3}));
  ASSERT_EQ(y.element_count(), 3);
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], -256.0f);
  EXPECT_FLOAT_EQ(py[1], -250.0f);
  EXPECT_FLOAT_EQ(py[2], 0.0f);
}

TEST(RunNodes, RunNodeDynamicQuantizeLinearFromDispatchTable) {
  RuntimeContext rt(KernelContext(DefaultOpset(13)));
  rt.tensors()["x"] = Tensor::FromFloat("x", {4}, {0.0f, 2.0f, -2.0f, 4.0f});

  NodeProto node = MakeNode("DynamicQuantizeLinear", {"x"}, {"y", "y_scale", "y_zero_point"});
  RunNode(node, rt);

  const Tensor &y = rt.tensors().at("y");
  const Tensor &y_scale = rt.tensors().at("y_scale");
  const Tensor &y_zp = rt.tensors().at("y_zero_point");
  EXPECT_EQ(y.shape, std::vector<int64_t>({4}));
  EXPECT_EQ(y.data_type, static_cast<int32_t>(core::runtime::DataType::UINT8));
  EXPECT_EQ(y_scale.data_type, static_cast<int32_t>(core::runtime::DataType::FLOAT));
  EXPECT_EQ(y_zp.data_type, static_cast<int32_t>(core::runtime::DataType::UINT8));
  EXPECT_TRUE(y_scale.shape.empty());
  EXPECT_TRUE(y_zp.shape.empty());
}

TEST(RunNodes, RunNodesOnRepeatedProtoFieldChain) {
  // Builds the small graph:  t = Mul(x, y);  out = Sub(t, z)
  // and runs it by building a free-standing ExecutionPlan over a
  // RepeatedProtoField<NodeProto> (mirroring how a caller would feed
  // ``graph.node()``) and driving it through a RuntimeSession.
  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.tensors()["x"] = Tensor::FromFloat("x", {2}, {1.0f, 2.0f});
  rt.tensors()["y"] = Tensor::FromFloat("y", {2}, {3.0f, 4.0f});
  rt.tensors()["z"] = Tensor::FromFloat("z", {2}, {0.5f, 0.25f});

  utils::RepeatedProtoField<NodeProto> nodes;
  *nodes.Add() = MakeNode("Mul", {"x", "y"}, {"t"});
  *nodes.Add() = MakeNode("Sub", {"t", "z"}, {"out"});

  core::runtime::ExecutionPlan plan(nodes, {});
  core::runtime::RuntimeSession session(plan);
  session.Run(rt);

  ASSERT_NE(rt.tensors().find("t"), rt.tensors().end());
  ASSERT_NE(rt.tensors().find("out"), rt.tensors().end());
  const float *out = rt.tensors()["out"].AsFloat();
  ASSERT_EQ(rt.tensors()["out"].element_count(), 2);
  EXPECT_FLOAT_EQ(out[0], 1.0f * 3.0f - 0.5f);
  EXPECT_FLOAT_EQ(out[1], 2.0f * 4.0f - 0.25f);
  EXPECT_EQ(rt.tensors()["out"].name, "out");
}

TEST(RunNodes, RunNodesOnIteratorRangeFromVector) {
  // Same graph, but the node list starts out as a std::vector<NodeProto>;
  // it is converted to a RepeatedProtoField<NodeProto> (which any container
  // of NodeProto converts to) before building the ExecutionPlan / session,
  // so any container works.
  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.tensors()["a"] = Tensor::FromFloat("a", {1}, {6.0f});
  rt.tensors()["b"] = Tensor::FromFloat("b", {1}, {2.0f});

  std::vector<NodeProto> nodes;
  nodes.push_back(MakeNode("Div", {"a", "b"}, {"q"})); // q = 3
  nodes.push_back(MakeNode("Neg", {"q"}, {"out"}));    // out = -3

  utils::RepeatedProtoField<NodeProto> node_field(nodes);
  core::runtime::ExecutionPlan plan(node_field, {});
  core::runtime::RuntimeSession session(plan);
  session.Run(rt);

  const float *out = rt.tensors()["out"].AsFloat();
  ASSERT_EQ(rt.tensors()["out"].element_count(), 1);
  EXPECT_FLOAT_EQ(out[0], -3.0f);
}

TEST(RunNodes, RunNodeEyeLikeUsesAttributes) {
  // Verify the EyeLike trampoline forwards both the ``k`` and ``dtype``
  // attributes to ``kernel::EyeLike``. Input shape is copied (2x3) and
  // the output uses dtype=INT64 (=7) with ones on the super-diagonal.
  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.tensors()["x"] = Tensor::FromFloat("x", {2, 3}, {0, 0, 0, 0, 0, 0});
  NodeProto node = MakeNode("EyeLike", {"x"}, {"y"});
  AttributeProto *attr_k = node.add_attribute();
  attr_k->set_name("k");
  attr_k->set_type(AttributeProto::AttributeType::INT);
  attr_k->set_i(1);
  AttributeProto *attr_dtype = node.add_attribute();
  attr_dtype->set_name("dtype");
  attr_dtype->set_type(AttributeProto::AttributeType::INT);
  attr_dtype->set_i(7); // INT64
  RunNode(node, rt);
  const Tensor &y = rt.tensors()["y"];
  EXPECT_EQ(y.shape, std::vector<int64_t>({2, 3}));
  EXPECT_EQ(y.data_type, 7);
  const int64_t *got = y.AsInt64();
  EXPECT_EQ(got[0], 0);
  EXPECT_EQ(got[1], 1);
  EXPECT_EQ(got[2], 0);
  EXPECT_EQ(got[3], 0);
  EXPECT_EQ(got[4], 0);
  EXPECT_EQ(got[5], 1);
}

TEST(RunNodes, RunNodeAffineGridUsesAttributes) {
  RuntimeContext rt(KernelContext(DefaultOpset(20)));
  rt.tensors()["theta"] =
      Tensor::FromFloat("theta", {1, 2, 3}, {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f});
  rt.tensors()["size"] = Tensor::FromInt64("size", {4}, {1, 1, 2, 2});

  NodeProto node = MakeNode("AffineGrid", {"theta", "size"}, {"grid"});
  AttributeProto *align_corners_attr = node.add_attribute();
  align_corners_attr->set_name("align_corners");
  align_corners_attr->set_type(AttributeProto::AttributeType::INT);
  align_corners_attr->set_i(1);

  RunNode(node, rt);

  const Tensor &grid = rt.tensors()["grid"];
  EXPECT_EQ(grid.shape, (std::vector<int64_t>{1, 2, 2, 2}));
  const float *got = grid.AsFloat();
  EXPECT_FLOAT_EQ(got[0], -1.0f);
  EXPECT_FLOAT_EQ(got[1], -1.0f);
  EXPECT_FLOAT_EQ(got[2], 1.0f);
  EXPECT_FLOAT_EQ(got[3], -1.0f);
  EXPECT_FLOAT_EQ(got[4], -1.0f);
  EXPECT_FLOAT_EQ(got[5], 1.0f);
  EXPECT_FLOAT_EQ(got[6], 1.0f);
  EXPECT_FLOAT_EQ(got[7], 1.0f);
}

TEST(RunNodes, RunNodeBatchNormalizationFromDispatchTable) {
  // Verifies that the ``BatchNormalization`` trampoline routes through the
  // dispatch table, picks up the ``epsilon`` attribute, and produces the
  // inference-mode output ``Y = (X - mean) / sqrt(var + epsilon) * scale + B``
  // for a 1x2x1x3 input with per-channel parameters.
  RuntimeContext rt(KernelContext(DefaultOpset(15)));
  rt.tensors()["x"] = Tensor::FromFloat("x", {1, 2, 1, 3}, {-1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f});
  rt.tensors()["scale"] = Tensor::FromFloat("scale", {2}, {1.0f, 1.5f});
  rt.tensors()["bias"] = Tensor::FromFloat("bias", {2}, {0.0f, 1.0f});
  rt.tensors()["mean"] = Tensor::FromFloat("mean", {2}, {0.0f, 3.0f});
  rt.tensors()["var"] = Tensor::FromFloat("var", {2}, {1.0f, 1.5f});

  NodeProto node = MakeNode("BatchNormalization", {"x", "scale", "bias", "mean", "var"}, {"y"});
  AttributeProto *epsilon_attr = node.add_attribute();
  epsilon_attr->set_name("epsilon");
  epsilon_attr->set_type(AttributeProto::AttributeType::FLOAT);
  epsilon_attr->set_f(1e-2f);

  RunNode(node, rt);

  const Tensor &y = rt.tensors()["y"];
  EXPECT_EQ(y.shape, (std::vector<int64_t>{1, 2, 1, 3}));
  ASSERT_EQ(y.element_count(), 6);
  const float *got = y.AsFloat();
  // Channel 0: (x - 0) / sqrt(1 + 1e-2) * 1 + 0
  const float inv0 = 1.0f / std::sqrt(1.0f + 1e-2f);
  EXPECT_FLOAT_EQ(got[0], -1.0f * inv0);
  EXPECT_FLOAT_EQ(got[1], 0.0f);
  EXPECT_FLOAT_EQ(got[2], 1.0f * inv0);
  // Channel 1: (x - 3) / sqrt(1.5 + 1e-2) * 1.5 + 1
  const float inv1 = 1.0f / std::sqrt(1.5f + 1e-2f);
  EXPECT_NEAR(got[3], (2.0f - 3.0f) * inv1 * 1.5f + 1.0f, 1e-6f);
  EXPECT_NEAR(got[4], (3.0f - 3.0f) * inv1 * 1.5f + 1.0f, 1e-6f);
  EXPECT_NEAR(got[5], (4.0f - 3.0f) * inv1 * 1.5f + 1.0f, 1e-6f);
}

TEST(RunNodes, RunNodeBatchNormalizationTrainingModeFromDispatchTable) {
  // Verifies that ``training_mode = 1`` routes through the dispatch table and
  // produces the three training-mode outputs (Y, running_mean, running_var).
  RuntimeContext rt(KernelContext(DefaultOpset(15)));
  rt.tensors()["x"] = Tensor::FromFloat("x", {1, 2, 1, 3}, {-1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f});
  rt.tensors()["scale"] = Tensor::FromFloat("scale", {2}, {1.0f, 1.5f});
  rt.tensors()["bias"] = Tensor::FromFloat("bias", {2}, {0.0f, 1.0f});
  rt.tensors()["mean"] = Tensor::FromFloat("mean", {2}, {0.5f, 2.0f});
  rt.tensors()["var"] = Tensor::FromFloat("var", {2}, {1.0f, 2.0f});

  NodeProto node = MakeNode("BatchNormalization", {"x", "scale", "bias", "mean", "var"},
                            {"y", "running_mean", "running_var"});
  AttributeProto *training_attr = node.add_attribute();
  training_attr->set_name("training_mode");
  training_attr->set_type(AttributeProto::AttributeType::INT);
  training_attr->set_i(1);

  RunNode(node, rt);

  const float momentum = 0.9f;
  const float saved_mean0 = 0.0f, saved_mean1 = 3.0f;
  const float saved_var = 2.0f / 3.0f;

  const Tensor &y = rt.tensors()["y"];
  EXPECT_EQ(y.shape, (std::vector<int64_t>{1, 2, 1, 3}));
  const float *got = y.AsFloat();
  const float inv = 1.0f / std::sqrt(saved_var + 1e-5f);
  EXPECT_NEAR(got[0], (-1.0f - saved_mean0) * inv, 1e-4f);
  EXPECT_NEAR(got[3], (2.0f - saved_mean1) * inv * 1.5f + 1.0f, 1e-4f);

  const Tensor &rm = rt.tensors()["running_mean"];
  const Tensor &rv = rt.tensors()["running_var"];
  EXPECT_EQ(rm.shape, (std::vector<int64_t>{2}));
  EXPECT_EQ(rv.shape, (std::vector<int64_t>{2}));
  const float *prm = rm.AsFloat();
  const float *prv = rv.AsFloat();
  EXPECT_NEAR(prm[0], 0.5f * momentum + saved_mean0 * (1.0f - momentum), 1e-5f);
  EXPECT_NEAR(prm[1], 2.0f * momentum + saved_mean1 * (1.0f - momentum), 1e-5f);
  EXPECT_NEAR(prv[0], 1.0f * momentum + saved_var * (1.0f - momentum), 1e-5f);
  EXPECT_NEAR(prv[1], 2.0f * momentum + saved_var * (1.0f - momentum), 1e-5f);
}

TEST(RunNodes, RunNodeImageDecoderFromDispatchTable) {
  // ``ImageDecoder`` was introduced in ONNX opset 20. The reference kernel
  // does not link an image-decoding library and returns the empty
  // ``(0, 0, C)`` matrix mandated by the schema; ``pixel_format`` drives
  // the channel count.
  RuntimeContext rt(KernelContext(DefaultOpset(20)));
  rt.tensors()["encoded_stream"] =
      Tensor::FromUint8("encoded_stream", {4}, std::vector<uint8_t>{0x00, 0x01, 0x02, 0x03});

  NodeProto node = MakeNode("ImageDecoder", {"encoded_stream"}, {"image"});
  AttributeProto *pixel_format_attr = node.add_attribute();
  pixel_format_attr->set_name("pixel_format");
  pixel_format_attr->set_type(AttributeProto::AttributeType::STRING);
  pixel_format_attr->set_s("Grayscale");

  RunNode(node, rt);

  const Tensor &image = rt.tensors()["image"];
  EXPECT_EQ(image.data_type, static_cast<int32_t>(core::runtime::DataType::UINT8));
  EXPECT_EQ(image.shape, (std::vector<int64_t>{0, 0, 1}));
  EXPECT_EQ(image.element_count(), 0);
  EXPECT_TRUE(image.data.empty());
}

TEST(RunNodes, RunNodeImageDecoderDefaultsToRGB) {
  // Without ``pixel_format`` attribute, the default ``"RGB"`` produces a
  // 3-channel empty image.
  RuntimeContext rt(KernelContext(DefaultOpset(20)));
  rt.tensors()["encoded_stream"] =
      Tensor::FromUint8("encoded_stream", {2}, std::vector<uint8_t>{0xAA, 0xBB});

  NodeProto node = MakeNode("ImageDecoder", {"encoded_stream"}, {"image"});
  RunNode(node, rt);

  const Tensor &image = rt.tensors()["image"];
  EXPECT_EQ(image.shape, (std::vector<int64_t>{0, 0, 3}));
}

TEST(RunNodes, RunNodeExpandFromDispatchTable) {
  // Expand broadcasts a (3, 1) input to a (3, 4) shape.
  RuntimeContext rt(KernelContext(DefaultOpset(13)));
  rt.tensors()["x"] = Tensor::FromFloat("x", {3, 1}, {1, 2, 3});
  rt.tensors()["shape"] = Tensor::FromInt64("shape", {2}, {3, 4});
  NodeProto node = MakeNode("Expand", {"x", "shape"}, {"y"});
  RunNode(node, rt);
  const Tensor &y = rt.tensors()["y"];
  EXPECT_EQ(y.shape, (std::vector<int64_t>{3, 4}));
  const float *got = y.AsFloat();
  EXPECT_FLOAT_EQ(got[0], 1.0f);
  EXPECT_FLOAT_EQ(got[3], 1.0f);
  EXPECT_FLOAT_EQ(got[4], 2.0f);
  EXPECT_FLOAT_EQ(got[11], 3.0f);
}

TEST(RunNodes, RunNodeTileFromDispatchTable) {
  // Tile repeats a (2, 2) input by (2, 2) to produce a (4, 4) output.
  RuntimeContext rt(KernelContext(DefaultOpset(13)));
  rt.tensors()["x"] = Tensor::FromFloat("x", {2, 2}, {0.0f, 1.0f, 2.0f, 3.0f});
  rt.tensors()["repeats"] = Tensor::FromInt64("repeats", {2}, {2, 2});
  NodeProto node = MakeNode("Tile", {"x", "repeats"}, {"y"});
  RunNode(node, rt);
  const Tensor &y = rt.tensors()["y"];
  EXPECT_EQ(y.shape, (std::vector<int64_t>{4, 4}));
  const float *got = y.AsFloat();
  const std::vector<float> expected = {0, 1, 0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 2, 3, 2, 3};
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_FLOAT_EQ(got[i], expected[i]);
  }
}

TEST(RunNodes, RunNodeReshapeFromDispatchTable) {
  // Reshape with two inputs (data, shape) and the default ``allowzero`` (0).
  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.tensors()["x"] = Tensor::FromFloat("x", {2, 3}, {1, 2, 3, 4, 5, 6});
  rt.tensors()["shape"] = Tensor::FromInt64("shape", {2}, {3, 2});
  NodeProto node = MakeNode("Reshape", {"x", "shape"}, {"y"});
  RunNode(node, rt);
  const Tensor &y = rt.tensors()["y"];
  EXPECT_EQ(y.shape, (std::vector<int64_t>{3, 2}));
  const float *got = y.AsFloat();
  EXPECT_FLOAT_EQ(got[0], 1.0f);
  EXPECT_FLOAT_EQ(got[5], 6.0f);
}

TEST(RunNodes, RunNodeResizeScalesFromDispatchTable) {
  // Resize via the (X, roi="", scales) input form. Upsamples a 1x1x2x2
  // NCHW tensor by [1, 1, 2, 3] using nearest mode and the asymmetric
  // coordinate transformation.
  RuntimeContext rt(KernelContext(DefaultOpset(13)));
  rt.tensors()["X"] = Tensor::FromFloat("X", {1, 1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  rt.tensors()["scales"] = Tensor::FromFloat("scales", {4}, {1.0f, 1.0f, 2.0f, 3.0f});
  NodeProto node = MakeNode("Resize", {"X", "", "scales"}, {"Y"});
  AttributeProto *coord = node.add_attribute();
  coord->set_name("coordinate_transformation_mode");
  coord->set_type(AttributeProto::AttributeType::STRING);
  coord->set_s("asymmetric");

  RunNode(node, rt);

  const Tensor &y = rt.tensors().at("Y");
  EXPECT_EQ(y.shape, (std::vector<int64_t>{1, 1, 4, 6}));
  EXPECT_EQ(y.data_type, static_cast<int32_t>(core::runtime::DataType::FLOAT));

  // Compare against the kernel's direct output to validate dispatch-time
  // wiring of inputs, attributes and outputs.
  onnx_kernels::kernel::Resize::Attributes attrs;
  attrs.coordinate_transformation_mode = "asymmetric";
  const onnx_kernels::kernel::Resize resize_kernel(rt.kernel_ctx());
  const Tensor y_ref = resize_kernel(rt.tensors().at("X"), rt.tensors().at("scales"), attrs);
  ASSERT_EQ(y.element_count(), y_ref.element_count());
  for (int64_t i = 0; i < y.element_count(); ++i) {
    EXPECT_FLOAT_EQ(y.AsFloat()[i], y_ref.AsFloat()[i]);
  }
}

TEST(RunNodes, RunNodeResizeSizesFromDispatchTable) {
  // Resize via the (X, roi="", scales="", sizes) input form.
  RuntimeContext rt(KernelContext(DefaultOpset(13)));
  rt.tensors()["X"] = Tensor::FromFloat("X", {1, 1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  rt.tensors()["sizes"] = Tensor::FromInt64("sizes", {4}, {1, 1, 4, 6});
  NodeProto node = MakeNode("Resize", {"X", "", "", "sizes"}, {"Y"});
  AttributeProto *coord = node.add_attribute();
  coord->set_name("coordinate_transformation_mode");
  coord->set_type(AttributeProto::AttributeType::STRING);
  coord->set_s("asymmetric");

  RunNode(node, rt);

  const Tensor &y = rt.tensors().at("Y");
  EXPECT_EQ(y.shape, (std::vector<int64_t>{1, 1, 4, 6}));
  EXPECT_EQ(y.data_type, static_cast<int32_t>(core::runtime::DataType::FLOAT));
}

TEST(RunNodes, RunNodeRegexFullMatchFromDispatchTable) {
  RuntimeContext rt(KernelContext(DefaultOpset(20)));
  rt.tensors()["x"] = Tensor::FromStrings("x", {3}, {"abc", "abcdef", "xyz"});
  NodeProto node = MakeNode("RegexFullMatch", {"x"}, {"y"});
  AttributeProto *pattern = node.add_attribute();
  pattern->set_name("pattern");
  pattern->set_type(AttributeProto::AttributeType::STRING);
  pattern->set_s("abc");
  RunNode(node, rt);
  const Tensor &y = rt.tensors()["y"];
  EXPECT_EQ(y.data_type, DataType::BOOL);
  EXPECT_EQ(y.shape, (std::vector<int64_t>{3}));
  const uint8_t *got = y.AsBool();
  ASSERT_NE(got, nullptr);
  EXPECT_EQ(got[0], 1u);
  EXPECT_EQ(got[1], 0u);
  EXPECT_EQ(got[2], 0u);
}

TEST(RunNodes, RunNodeGatherFromDispatchTable) {
  // Gather along ``axis=0`` selects whole rows from ``data``.
  RuntimeContext rt(KernelContext(DefaultOpset(13)));
  rt.tensors()["data"] = Tensor::FromFloat("data", {3, 2}, {1, 2, 3, 4, 5, 6});
  rt.tensors()["indices"] = Tensor::FromInt64("indices", {2}, {2, 0});
  NodeProto node = MakeNode("Gather", {"data", "indices"}, {"y"});
  AttributeProto *axis = node.add_attribute();
  axis->set_name("axis");
  axis->set_type(AttributeProto::AttributeType::INT);
  axis->set_i(0);
  RunNode(node, rt);
  const Tensor &y = rt.tensors()["y"];
  EXPECT_EQ(y.shape, (std::vector<int64_t>{2, 2}));
  const float *got = y.AsFloat();
  EXPECT_FLOAT_EQ(got[0], 5.0f);
  EXPECT_FLOAT_EQ(got[1], 6.0f);
  EXPECT_FLOAT_EQ(got[2], 1.0f);
  EXPECT_FLOAT_EQ(got[3], 2.0f);
}

TEST(RunNodes, RunNodeGatherFromDispatchTableDefaultAxis) {
  // Default ``axis`` is 0; verify the dispatch entry handles a missing attribute.
  RuntimeContext rt(KernelContext(DefaultOpset(13)));
  rt.tensors()["data"] = Tensor::FromFloat("data", {3}, {10, 20, 30});
  rt.tensors()["indices"] = Tensor::FromInt64("indices", {2}, {1, 2});
  NodeProto node = MakeNode("Gather", {"data", "indices"}, {"y"});
  RunNode(node, rt);
  const Tensor &y = rt.tensors()["y"];
  EXPECT_EQ(y.shape, (std::vector<int64_t>{2}));
  const float *got = y.AsFloat();
  EXPECT_FLOAT_EQ(got[0], 20.0f);
  EXPECT_FLOAT_EQ(got[1], 30.0f);
}

TEST(RunNodes, RunNodeGatherNDFromDispatchTable) {
  // GatherND with ``batch_dims=0`` picks scalars from a 2-D ``data`` tensor.
  RuntimeContext rt(KernelContext(DefaultOpset(13)));
  rt.tensors()["data"] = Tensor::FromFloat("data", {2, 2}, {1, 2, 3, 4});
  rt.tensors()["indices"] = Tensor::FromInt64("indices", {2, 2}, {0, 0, 1, 1});
  NodeProto node = MakeNode("GatherND", {"data", "indices"}, {"y"});
  RunNode(node, rt);
  const Tensor &y = rt.tensors()["y"];
  EXPECT_EQ(y.shape, (std::vector<int64_t>{2}));
  const float *got = y.AsFloat();
  EXPECT_FLOAT_EQ(got[0], 1.0f);
  EXPECT_FLOAT_EQ(got[1], 4.0f);
}

TEST(RunNodes, RunNodeGatherNDWithBatchDimsFromDispatchTable) {
  // GatherND with ``batch_dims=1`` independently indexes each batch row.
  RuntimeContext rt(KernelContext(DefaultOpset(13)));
  rt.tensors()["data"] =
      Tensor::FromFloat("data", {2, 3, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  rt.tensors()["indices"] = Tensor::FromInt64("indices", {2, 1}, {1, 0});
  NodeProto node = MakeNode("GatherND", {"data", "indices"}, {"y"});
  AttributeProto *attr = node.add_attribute();
  attr->set_name("batch_dims");
  attr->set_type(AttributeProto::AttributeType::INT);
  attr->set_i(1);
  RunNode(node, rt);
  const Tensor &y = rt.tensors()["y"];
  EXPECT_EQ(y.shape, (std::vector<int64_t>{2, 2}));
  const float *got = y.AsFloat();
  EXPECT_FLOAT_EQ(got[0], 3.0f);
  EXPECT_FLOAT_EQ(got[1], 4.0f);
  EXPECT_FLOAT_EQ(got[2], 7.0f);
  EXPECT_FLOAT_EQ(got[3], 8.0f);
}

TEST(RunNodes, RunNodeSqueezeAxesAsInput) {
  // Opset 13+: ``axes`` is provided as the optional second INT64 input.
  RuntimeContext rt(KernelContext(DefaultOpset(13)));
  rt.tensors()["x"] = Tensor::FromFloat("x", {1, 3, 1, 2}, {1, 2, 3, 4, 5, 6});
  rt.tensors()["axes"] = Tensor::FromInt64("axes", {2}, {0, 2});
  NodeProto node = MakeNode("Squeeze", {"x", "axes"}, {"y"});
  RunNode(node, rt);
  const Tensor &y = rt.tensors()["y"];
  EXPECT_EQ(y.shape, (std::vector<int64_t>{3, 2}));
  EXPECT_EQ(y.element_count(), 6);
}

TEST(RunNodes, RunNodeSqueezeAxesAsAttribute) {
  // Opset <13: ``axes`` is an INTS attribute (also accepted by the trampoline
  // for backward compatibility).
  RuntimeContext rt(KernelContext(DefaultOpset(11)));
  rt.tensors()["x"] = Tensor::FromFloat("x", {1, 3, 1, 2}, {1, 2, 3, 4, 5, 6});
  NodeProto node = MakeNode("Squeeze", {"x"}, {"y"});
  AttributeProto *attr = node.add_attribute();
  attr->set_name("axes");
  attr->set_type(AttributeProto::AttributeType::INTS);
  attr->add_ints(static_cast<int64_t>(0));
  attr->add_ints(static_cast<int64_t>(2));
  RunNode(node, rt);
  const Tensor &y = rt.tensors()["y"];
  EXPECT_EQ(y.shape, (std::vector<int64_t>{3, 2}));
}

TEST(RunNodes, RunNodeSqueezeAxesMissingInput) {
  // Opset 13+: ``axes`` is an optional input. When the node is declared with
  // only the ``data`` input (the optional ``axes`` slot is omitted entirely),
  // the kernel squeezes every dimension equal to 1, matching the ONNX spec.
  RuntimeContext rt(KernelContext(DefaultOpset(13)));
  rt.tensors()["x"] = Tensor::FromFloat("x", {1, 3, 1, 2}, {1, 2, 3, 4, 5, 6});
  NodeProto node = MakeNode("Squeeze", {"x"}, {"y"});
  RunNode(node, rt);
  const Tensor &y = rt.tensors()["y"];
  EXPECT_EQ(y.shape, (std::vector<int64_t>{3, 2}));
  EXPECT_EQ(y.element_count(), 6);
}

TEST(RunNodes, RunNodeSqueezeAxesEmptyName) {
  // Opset 13+: a missing optional input is conventionally encoded as an empty
  // input name. The kernel must treat this the same as omitting the slot
  // entirely and squeeze all unit dimensions.
  RuntimeContext rt(KernelContext(DefaultOpset(13)));
  rt.tensors()["x"] = Tensor::FromFloat("x", {1, 3, 1, 2}, {1, 2, 3, 4, 5, 6});
  NodeProto node = MakeNode("Squeeze", {"x", ""}, {"y"});
  RunNode(node, rt);
  const Tensor &y = rt.tensors()["y"];
  EXPECT_EQ(y.shape, (std::vector<int64_t>{3, 2}));
  EXPECT_EQ(y.element_count(), 6);
}

TEST(RunNodes, RunNodeUnsqueezeAxesAsInput) {
  RuntimeContext rt(KernelContext(DefaultOpset(13)));
  rt.tensors()["x"] = Tensor::FromFloat("x", {3, 2}, {1, 2, 3, 4, 5, 6});
  rt.tensors()["axes"] = Tensor::FromInt64("axes", {2}, {0, 2});
  NodeProto node = MakeNode("Unsqueeze", {"x", "axes"}, {"y"});
  RunNode(node, rt);
  const Tensor &y = rt.tensors()["y"];
  EXPECT_EQ(y.shape, (std::vector<int64_t>{1, 3, 1, 2}));
}

TEST(RunNodes, RunNodeUnsqueezeScalarAxesInput) {
  // Some upstream function bodies (e.g. ai.onnx::AffineGrid) feed the
  // Unsqueeze ``axes`` input with a 0-D INT64 scalar instead of the 1-D
  // tensor required by the schema. For compatibility with those models
  // (which the upstream reference evaluator also accepts) the trampoline
  // tolerates a rank-0 axes tensor.
  RuntimeContext rt(KernelContext(DefaultOpset(13)));
  rt.tensors()["x"] = Tensor::FromFloat("x", {3, 2}, {1, 2, 3, 4, 5, 6});
  rt.tensors()["axes"] = Tensor::FromInt64("axes", {}, {-1});
  NodeProto node = MakeNode("Unsqueeze", {"x", "axes"}, {"y"});
  RunNode(node, rt);
  const Tensor &y = rt.tensors()["y"];
  EXPECT_EQ(y.shape, (std::vector<int64_t>{3, 2, 1}));
}

TEST(RunNodes, RunNodeShapeNoAttributes) {
  // Default attributes: returns the full shape as an INT64 1-D tensor.
  RuntimeContext rt(KernelContext(DefaultOpset(15)));
  rt.tensors()["x"] = Tensor::FromFloat("x", {2, 3, 4}, std::vector<float>(24, 0.0f));
  NodeProto node = MakeNode("Shape", {"x"}, {"y"});
  RunNode(node, rt);
  const Tensor &y = rt.tensors()["y"];
  EXPECT_EQ(y.shape, (std::vector<int64_t>{3}));
  EXPECT_EQ(y.data_type, 7); // INT64
  const int64_t *got = y.AsInt64();
  EXPECT_EQ(got[0], 2);
  EXPECT_EQ(got[1], 3);
  EXPECT_EQ(got[2], 4);
}

TEST(RunNodes, RunNodeShapeUsesStartAndEndAttributes) {
  // Verify the Shape trampoline forwards the ``start`` and ``end``
  // attributes to ``kernel::Shape``: ``shape[1:-1]`` of a 4-D input.
  RuntimeContext rt(KernelContext(DefaultOpset(15)));
  rt.tensors()["x"] = Tensor::FromFloat("x", {2, 3, 4, 5}, std::vector<float>(120, 0.0f));
  NodeProto node = MakeNode("Shape", {"x"}, {"y"});
  AttributeProto *attr_start = node.add_attribute();
  attr_start->set_name("start");
  attr_start->set_type(AttributeProto::AttributeType::INT);
  attr_start->set_i(1);
  AttributeProto *attr_end = node.add_attribute();
  attr_end->set_name("end");
  attr_end->set_type(AttributeProto::AttributeType::INT);
  attr_end->set_i(-1);
  RunNode(node, rt);
  const Tensor &y = rt.tensors()["y"];
  EXPECT_EQ(y.shape, (std::vector<int64_t>{2}));
  EXPECT_EQ(y.data_type, 7); // INT64
  const int64_t *got = y.AsInt64();
  EXPECT_EQ(got[0], 3);
  EXPECT_EQ(got[1], 4);
}

TEST(RunNodes, RunNodeEinsumUsesEquationAttribute) {
  // Verify the Einsum trampoline forwards the variadic inputs and the
  // ``equation`` STRING attribute to ``kernel::Einsum``. The equation
  // ``"ij,jk->ik"`` performs a 2x3 by 3x2 matrix product.
  RuntimeContext rt(KernelContext(DefaultOpset(13)));
  rt.tensors()["a"] = Tensor::FromFloat("a", {2, 3}, {1, 2, 3, 4, 5, 6});
  rt.tensors()["b"] = Tensor::FromFloat("b", {3, 2}, {7, 8, 9, 10, 11, 12});
  NodeProto node = MakeNode("Einsum", {"a", "b"}, {"y"});
  AttributeProto *attr = node.add_attribute();
  attr->set_name("equation");
  attr->set_type(AttributeProto::AttributeType::STRING);
  attr->set_s("ij,jk->ik");
  RunNode(node, rt);
  const Tensor &y = rt.tensors()["y"];
  EXPECT_EQ(y.shape, (std::vector<int64_t>{2, 2}));
  const float *got = y.AsFloat();
  EXPECT_FLOAT_EQ(got[0], 58.0f);
  EXPECT_FLOAT_EQ(got[1], 64.0f);
  EXPECT_FLOAT_EQ(got[2], 139.0f);
  EXPECT_FLOAT_EQ(got[3], 154.0f);
}

TEST(RunNodes, RunNodeDFTOpset17UsesAxisAttribute) {
  // v17 DFT: ``axis`` / ``inverse`` / ``onesided`` are INT attributes.
  // Inputs are ``(input, dft_length?)``.
  RuntimeContext rt(KernelContext(DefaultOpset(17)));
  Tensor x = Tensor::FromFloat("x", {1, 4, 1}, {1.0f, 2.0f, 3.0f, 4.0f});
  rt.tensors()["x"] = x;

  NodeProto node = MakeNode("DFT", {"x"}, {"y"});
  AttributeProto *axis = node.add_attribute();
  axis->set_name("axis");
  axis->set_type(AttributeProto::AttributeType::INT);
  axis->set_i(1);

  RunNode(node, rt);

  onnx_kernels::kernel::DFT ref(rt.kernel_ctx());
  Tensor expected = ref(x, /*dft_length=*/nullptr, /*axis=*/1, /*onesided=*/false,
                        /*inverse=*/false);
  const Tensor &y = rt.tensors()["y"];
  ASSERT_EQ(y.shape, expected.shape);
  ASSERT_EQ(y.data.size(), expected.data.size());
  EXPECT_EQ(std::memcmp(y.data.data(), expected.data.data(), expected.data.size()), 0);
}

TEST(RunNodes, RunNodeDFTOpset20UsesAxisInput) {
  // v20 DFT: ``axis`` becomes the third (optional) input; only
  // ``inverse`` / ``onesided`` remain attributes.
  RuntimeContext rt(KernelContext(DefaultOpset(20)));
  Tensor x = Tensor::FromFloat("x", {1, 4, 1}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor axis = Tensor::FromInt64("axis", {}, {1});
  rt.tensors()["x"] = x;
  rt.tensors()["axis"] = axis;

  // ``dft_length`` is omitted by passing an empty input name.
  NodeProto node = MakeNode("DFT", {"x", "", "axis"}, {"y"});

  RunNode(node, rt);

  onnx_kernels::kernel::DFT ref(rt.kernel_ctx());
  Tensor expected = ref(x, /*dft_length=*/nullptr, /*axis=*/1, /*onesided=*/false,
                        /*inverse=*/false);
  const Tensor &y = rt.tensors()["y"];
  ASSERT_EQ(y.shape, expected.shape);
  ASSERT_EQ(y.data.size(), expected.data.size());
  EXPECT_EQ(std::memcmp(y.data.data(), expected.data.data(), expected.data.size()), 0);
}

TEST(RunNodes, RunNodeDFTOpset17InverseOnesidedAttributes) {
  // v17 DFT with ``inverse`` and ``onesided`` attributes set.
  RuntimeContext rt(KernelContext(DefaultOpset(17)));
  Tensor x = Tensor::FromFloat("x", {1, 4, 2}, {1.0f, 0.0f, 2.0f, 0.0f, 3.0f, 0.0f, 4.0f, 0.0f});
  rt.tensors()["x"] = x;

  NodeProto node = MakeNode("DFT", {"x"}, {"y"});
  AttributeProto *axis = node.add_attribute();
  axis->set_name("axis");
  axis->set_type(AttributeProto::AttributeType::INT);
  axis->set_i(1);
  AttributeProto *inverse = node.add_attribute();
  inverse->set_name("inverse");
  inverse->set_type(AttributeProto::AttributeType::INT);
  inverse->set_i(1);

  RunNode(node, rt);

  onnx_kernels::kernel::DFT ref(rt.kernel_ctx());
  Tensor expected = ref(x, /*dft_length=*/nullptr, /*axis=*/1, /*onesided=*/false,
                        /*inverse=*/true);
  const Tensor &y = rt.tensors()["y"];
  ASSERT_EQ(y.shape, expected.shape);
  ASSERT_EQ(y.data.size(), expected.data.size());
  EXPECT_EQ(std::memcmp(y.data.data(), expected.data.data(), expected.data.size()), 0);
}

TEST(RunNodes, RunNodeAdagradFromDispatchTable) {
  // Dispatches a single-tensor Adagrad node and checks the outputs
  // against ``kernel::Adagrad`` invoked directly.
  RuntimeContext rt(KernelContext(DefaultOpset(13)));
  rt.tensors()["R"] = Tensor::FromFloat("R", {}, {0.1f});
  rt.tensors()["T"] = Tensor::FromInt64("T", {}, {0});
  rt.tensors()["X"] = Tensor::FromFloat("X", {2}, {1.0f, 2.0f});
  rt.tensors()["G"] = Tensor::FromFloat("G", {2}, {-0.5f, 0.25f});
  rt.tensors()["H"] = Tensor::FromFloat("H", {2}, {0.1f, 0.1f});

  NodeProto node = MakeNode("Adagrad", {"R", "T", "X", "G", "H"}, {"X_new", "H_new"},
                            "ai.onnx.preview.training");
  AttributeProto *eps = node.add_attribute();
  eps->set_name("epsilon");
  eps->set_type(AttributeProto::AttributeType::FLOAT);
  eps->set_f(1e-5f);
  AttributeProto *nc = node.add_attribute();
  nc->set_name("norm_coefficient");
  nc->set_type(AttributeProto::AttributeType::FLOAT);
  nc->set_f(0.001f);

  RunNode(node, rt);

  onnx_kernels::kernel::Adagrad ref(rt.kernel_ctx());
  std::vector<Tensor> expected = ref(rt.tensors()["R"], rt.tensors()["T"], {rt.tensors()["X"]},
                                     {rt.tensors()["G"]}, {rt.tensors()["H"]}, 1e-5f, 0.0f, 0.001f);

  const Tensor &x_new = rt.tensors()["X_new"];
  const Tensor &h_new = rt.tensors()["H_new"];
  ASSERT_EQ(x_new.shape, (std::vector<int64_t>{2}));
  ASSERT_EQ(h_new.shape, (std::vector<int64_t>{2}));
  EXPECT_FLOAT_EQ(x_new.AsFloat()[0], expected[0].AsFloat()[0]);
  EXPECT_FLOAT_EQ(x_new.AsFloat()[1], expected[0].AsFloat()[1]);
  EXPECT_FLOAT_EQ(h_new.AsFloat()[0], expected[1].AsFloat()[0]);
  EXPECT_FLOAT_EQ(h_new.AsFloat()[1], expected[1].AsFloat()[1]);
}

TEST(RunNodes, RunNodeMomentumNesterovFromDispatchTable) {
  RuntimeContext rt(KernelContext(DefaultOpset(13)));
  rt.tensors()["R"] = Tensor::FromFloat("R", {}, {0.1f});
  rt.tensors()["T"] = Tensor::FromInt64("T", {}, {0});
  rt.tensors()["X"] = Tensor::FromFloat("X", {2}, {1.2f, 2.8f});
  rt.tensors()["G"] = Tensor::FromFloat("G", {2}, {-0.94f, -2.5f});
  rt.tensors()["V"] = Tensor::FromFloat("V", {2}, {1.7f, 3.6f});

  NodeProto node = MakeNode("Momentum", {"R", "T", "X", "G", "V"}, {"X_new", "V_new"},
                            "ai.onnx.preview.training");
  AttributeProto *a = node.add_attribute();
  a->set_name("alpha");
  a->set_type(AttributeProto::AttributeType::FLOAT);
  a->set_f(0.95f);
  AttributeProto *b = node.add_attribute();
  b->set_name("beta");
  b->set_type(AttributeProto::AttributeType::FLOAT);
  b->set_f(1.0f);
  AttributeProto *nc = node.add_attribute();
  nc->set_name("norm_coefficient");
  nc->set_type(AttributeProto::AttributeType::FLOAT);
  nc->set_f(0.01f);
  AttributeProto *mode = node.add_attribute();
  mode->set_name("mode");
  mode->set_type(AttributeProto::AttributeType::STRING);
  mode->set_s("nesterov");

  RunNode(node, rt);

  onnx_kernels::kernel::Momentum ref(rt.kernel_ctx());
  std::vector<Tensor> expected =
      ref(rt.tensors()["R"], rt.tensors()["T"], {rt.tensors()["X"]}, {rt.tensors()["G"]},
          {rt.tensors()["V"]}, 0.95f, 1.0f, 0.01f, onnx_kernels::kernel::Momentum::Mode::kNesterov);

  const Tensor &x_new = rt.tensors()["X_new"];
  const Tensor &v_new = rt.tensors()["V_new"];
  EXPECT_FLOAT_EQ(x_new.AsFloat()[0], expected[0].AsFloat()[0]);
  EXPECT_FLOAT_EQ(x_new.AsFloat()[1], expected[0].AsFloat()[1]);
  EXPECT_FLOAT_EQ(v_new.AsFloat()[0], expected[1].AsFloat()[0]);
  EXPECT_FLOAT_EQ(v_new.AsFloat()[1], expected[1].AsFloat()[1]);
}

TEST(RunNodes, RunNodeAdamMultipleFromDispatchTable) {
  // Dispatches a two-tensor Adam node (N=2) and checks the 3*N=6
  // outputs against the direct kernel invocation.
  RuntimeContext rt(KernelContext(DefaultOpset(13)));
  rt.tensors()["R"] = Tensor::FromFloat("R", {}, {0.05f});
  rt.tensors()["T"] = Tensor::FromInt64("T", {}, {5});
  rt.tensors()["X1"] = Tensor::FromFloat("X1", {2}, {0.5f, -0.5f});
  rt.tensors()["X2"] = Tensor::FromFloat("X2", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  rt.tensors()["G1"] = Tensor::FromFloat("G1", {2}, {0.1f, -0.2f});
  rt.tensors()["G2"] = Tensor::FromFloat("G2", {2, 2}, {-0.5f, 0.25f, 0.75f, -1.0f});
  rt.tensors()["V1"] = Tensor::FromFloat("V1", {2}, {0.01f, 0.02f});
  rt.tensors()["V2"] = Tensor::FromFloat("V2", {2, 2}, {0.05f, 0.05f, -0.05f, 0.0f});
  rt.tensors()["H1"] = Tensor::FromFloat("H1", {2}, {0.001f, 0.002f});
  rt.tensors()["H2"] = Tensor::FromFloat("H2", {2, 2}, {0.01f, 0.02f, 0.03f, 0.04f});

  NodeProto node = MakeNode("Adam", {"R", "T", "X1", "X2", "G1", "G2", "V1", "V2", "H1", "H2"},
                            {"X1n", "X2n", "V1n", "V2n", "H1n", "H2n"}, "ai.onnx.preview.training");
  for (const auto &kv :
       std::vector<std::pair<std::string, float>>{{"alpha", 0.9f},
                                                  {"beta", 0.999f},
                                                  {"epsilon", 1e-6f},
                                                  {"norm_coefficient", 0.0f},
                                                  {"norm_coefficient_post", 0.0f}}) {
    AttributeProto *a = node.add_attribute();
    a->set_name(kv.first.c_str());
    a->set_type(AttributeProto::AttributeType::FLOAT);
    a->set_f(kv.second);
  }

  RunNode(node, rt);

  onnx_kernels::kernel::Adam ref(rt.kernel_ctx());
  std::vector<Tensor> expected =
      ref(rt.tensors()["R"], rt.tensors()["T"], {rt.tensors()["X1"], rt.tensors()["X2"]},
          {rt.tensors()["G1"], rt.tensors()["G2"]}, {rt.tensors()["V1"], rt.tensors()["V2"]},
          {rt.tensors()["H1"], rt.tensors()["H2"]}, 0.9f, 0.999f, 1e-6f, 0.0f, 0.0f);

  const std::vector<std::string> out_names = {"X1n", "X2n", "V1n", "V2n", "H1n", "H2n"};
  for (size_t i = 0; i < out_names.size(); ++i) {
    const Tensor &got = rt.tensors()[out_names[i]];
    ASSERT_EQ(got.shape, expected[i].shape);
    ASSERT_EQ(got.element_count(), expected[i].element_count());
    for (int64_t j = 0; j < got.element_count(); ++j) {
      EXPECT_FLOAT_EQ(got.AsFloat()[j], expected[i].AsFloat()[j])
          << " at out=" << out_names[i] << " idx=" << j;
    }
  }
}

TEST(RunNodes, RunNodeMomentumRejectsBadInputCount) {
  RuntimeContext rt(KernelContext(DefaultOpset(13)));
  rt.tensors()["R"] = Tensor::FromFloat("R", {}, {0.1f});
  rt.tensors()["T"] = Tensor::FromInt64("T", {}, {0});
  rt.tensors()["X"] = Tensor::FromFloat("X", {1}, {1.0f});
  rt.tensors()["G"] = Tensor::FromFloat("G", {1}, {1.0f});
  // 2 + 2 = 4 inputs, not 2 + 3*N: should throw.
  NodeProto node =
      MakeNode("Momentum", {"R", "T", "X", "G"}, {"X_new"}, "ai.onnx.preview.training");
  EXPECT_THROW(RunNode(node, rt), std::invalid_argument);
}

TEST(RunNodes, RunNodeUnsupportedOpTypeThrows) {
  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.tensors()["x"] = Tensor::FromFloat("x", {1}, {1.0f});
  NodeProto node = MakeNode("ThisOpDoesNotExist", {"x"}, {"y"});
  EXPECT_THROW(RunNode(node, rt), std::invalid_argument);
}

TEST(RunNodes, RunNodeMissingInputThrows) {
  RuntimeContext rt(KernelContext(DefaultOpset(18))); // empty: "x" is not present
  NodeProto node = MakeNode("Abs", {"x"}, {"y"});
  EXPECT_THROW(RunNode(node, rt), std::invalid_argument);
}

TEST(RunNodes, RunNodeWrongInputCountThrows) {
  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.tensors()["x"] = Tensor::FromFloat("x", {1}, {1.0f});
  // Add expects two inputs but we only declare one.
  NodeProto node = MakeNode("Add", {"x"}, {"y"});
  EXPECT_THROW(RunNode(node, rt), std::invalid_argument);
}

TEST(RunNodes, RuntimeContextSetGetHas) {
  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  EXPECT_FALSE(rt.Has("x"));
  rt.Set("x", Tensor::FromFloat("x", {1}, {7.0f}));
  EXPECT_TRUE(rt.Has("x"));
  EXPECT_FLOAT_EQ(rt.Get("x").AsFloat()[0], 7.0f);
  EXPECT_THROW(rt.Get("missing"), std::out_of_range);
}

TEST(RunNodes, RuntimeContextRemove) {
  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("x", Tensor::FromFloat("x", {1}, {7.0f}));
  EXPECT_TRUE(rt.Remove("x"));
  EXPECT_FALSE(rt.Has("x"));
  EXPECT_THROW(rt.Get("x"), std::out_of_range);
  EXPECT_FALSE(rt.Remove("x"));
}

TEST(RunNodes, RuntimeContextEventLogSetReplaceRemove) {
  using core::runtime::RuntimeEventAction;
  using core::runtime::RuntimeEventKind;
  RuntimeContext rt(KernelContext(DefaultOpset(18)),
                    core::runtime::RuntimeContextOptions{.events_enabled = true});
  EXPECT_TRUE(rt.events().empty());

  // Set -> add event with values populated (element_count <= 8). Default
  // kind for Set is "input".
  rt.Set("x", Tensor::FromFloat("x", {3}, {1.0f, -2.0f, 3.5f}));
  ASSERT_EQ(rt.events().size(), 1u);
  const auto &add_ev = rt.events()[0];
  EXPECT_EQ(add_ev.action, RuntimeEventAction::kAdd);
  EXPECT_EQ(add_ev.kind, RuntimeEventKind::kInput);
  EXPECT_EQ(add_ev.name, "x");
  // Inputs report node_index = -1 and the CPU device (-1).
  EXPECT_EQ(add_ev.node_index, -1);
  EXPECT_EQ(add_ev.device, -1);
  EXPECT_EQ(add_ev.data_type, static_cast<int32_t>(DataType::FLOAT));
  EXPECT_EQ(add_ev.shape, (std::vector<int64_t>{3}));
  EXPECT_EQ(add_ev.value_count, 3);
  EXPECT_FLOAT_EQ(static_cast<float>(add_ev.values[0]), 1.0f);
  EXPECT_FLOAT_EQ(static_cast<float>(add_ev.values[1]), -2.0f);
  EXPECT_FLOAT_EQ(static_cast<float>(add_ev.values[2]), 3.5f);
  // Unused slots of the fixed-size buffer stay zero-initialised.
  EXPECT_DOUBLE_EQ(add_ev.values[3], 0.0);
  EXPECT_DOUBLE_EQ(add_ev.values[7], 0.0);
  EXPECT_GT(add_ev.timestamp_ns, 0);

  // Put on the same name -> replace event. Default kind for Put is
  // "intermediate".
  rt.Put("x", Tensor::FromInt32("x", {2}, {7, 8}));
  ASSERT_EQ(rt.events().size(), 2u);
  EXPECT_EQ(rt.events()[1].action, RuntimeEventAction::kReplace);
  EXPECT_EQ(rt.events()[1].kind, RuntimeEventKind::kIntermediate);
  EXPECT_EQ(rt.events()[1].data_type, static_cast<int32_t>(DataType::INT32));
  EXPECT_EQ(rt.events()[1].value_count, 2);
  EXPECT_DOUBLE_EQ(rt.events()[1].values[0], 7.0);
  EXPECT_DOUBLE_EQ(rt.events()[1].values[1], 8.0);

  // Remove -> remove event with empty shape and value_count = 0.
  EXPECT_TRUE(rt.Remove("x"));
  ASSERT_EQ(rt.events().size(), 3u);
  EXPECT_EQ(rt.events()[2].action, RuntimeEventAction::kRemove);
  EXPECT_EQ(rt.events()[2].name, "x");
  EXPECT_TRUE(rt.events()[2].shape.empty());
  EXPECT_EQ(rt.events()[2].value_count, 0);

  // No-op remove -> no extra event.
  EXPECT_FALSE(rt.Remove("x"));
  EXPECT_EQ(rt.events().size(), 3u);

  // Large tensor (> kRuntimeEventValueLimit) -> data_type = -1, empty shape,
  // values truncated to the first kRuntimeEventValueLimit entries.
  rt.Put("big", Tensor::FromInt32("big", {9}, {0, 1, 2, 3, 4, 5, 6, 7, 8}));
  ASSERT_EQ(rt.events().size(), 4u);
  EXPECT_EQ(rt.events()[3].action, RuntimeEventAction::kAdd);
  EXPECT_EQ(rt.events()[3].data_type, -1);
  EXPECT_TRUE(rt.events()[3].shape.empty());
  EXPECT_EQ(rt.events()[3].value_count, 8);
  for (int32_t i = 0; i < 8; ++i) {
    EXPECT_DOUBLE_EQ(rt.events()[3].values[i], static_cast<double>(i));
  }

  // String tensor values land in string_values (numeric values buffer stays
  // zero-initialised).
  rt.Put("s", Tensor::FromStrings("s", {2}, {"a", "bc"}));
  ASSERT_EQ(rt.events().size(), 5u);
  EXPECT_EQ(rt.events()[4].data_type, static_cast<int32_t>(DataType::STRING));
  EXPECT_EQ(rt.events()[4].value_count, 2);
  EXPECT_EQ(rt.events()[4].string_values[0], "a");
  EXPECT_EQ(rt.events()[4].string_values[1], "bc");
  EXPECT_DOUBLE_EQ(rt.events()[4].values[0], 0.0);

  rt.ClearEvents();
  EXPECT_TRUE(rt.events().empty());
}

TEST(RunNodes, RuntimeContextEventLogCapturesRunGraphMutations) {
  // Smoke test: running a small chain of nodes through the dispatcher
  // populates the event log via SetOutput / Put on every produced tensor
  // and also appends one ``kRunNode`` event per dispatched node.
  using core::runtime::RuntimeEventAction;
  using core::runtime::RuntimeEventKind;
  RuntimeContext rt(KernelContext(DefaultOpset(18)),
                    core::runtime::RuntimeContextOptions{.events_enabled = true});
  rt.Set("x", Tensor::FromFloat("x", {2}, {-1.0f, 2.0f}));
  rt.Set("z", Tensor::FromFloat("z", {2}, {10.0f, 20.0f}));
  rt.ClearEvents();

  std::vector<NodeProto> nodes;
  nodes.push_back(MakeNode("Abs", {"x"}, {"t"}));
  nodes.push_back(MakeNode("Add", {"t", "z"}, {"y"}));
  utils::RepeatedProtoField<NodeProto> node_field(nodes);
  core::runtime::ExecutionPlan plan(node_field, {});
  core::runtime::RuntimeSession session(plan);
  session.Run(rt);

  // Each node produces one tensor ``add`` event tagged as an
  // intermediate plus one ``run_node`` event summarising the dispatch.
  ASSERT_EQ(rt.events().size(), 4u);
  EXPECT_EQ(rt.events()[0].name, "t");
  EXPECT_EQ(rt.events()[0].action, RuntimeEventAction::kAdd);
  EXPECT_EQ(rt.events()[0].kind, RuntimeEventKind::kIntermediate);
  EXPECT_EQ(rt.events()[0].node_index, 0);
  EXPECT_EQ(rt.events()[0].device, -1);
  EXPECT_EQ(rt.events()[1].action, RuntimeEventAction::kRunNode);
  EXPECT_EQ(rt.events()[1].op_domain, "ai.onnx");
  EXPECT_EQ(rt.events()[1].op_type, "Abs");
  EXPECT_EQ(rt.events()[1].inputs, (std::vector<std::string>{"x"}));
  EXPECT_GE(rt.events()[1].duration_ns, 0);
  EXPECT_EQ(rt.events()[1].node_index, 0);
  EXPECT_EQ(rt.events()[2].name, "y");
  EXPECT_EQ(rt.events()[2].action, RuntimeEventAction::kAdd);
  EXPECT_EQ(rt.events()[2].kind, RuntimeEventKind::kIntermediate);
  EXPECT_EQ(rt.events()[2].node_index, 1);
  EXPECT_EQ(rt.events()[3].action, RuntimeEventAction::kRunNode);
  EXPECT_EQ(rt.events()[3].op_domain, "ai.onnx");
  EXPECT_EQ(rt.events()[3].op_type, "Add");
  EXPECT_EQ(rt.events()[3].inputs, (std::vector<std::string>{"t", "z"}));
  EXPECT_GE(rt.events()[3].duration_ns, 0);
  EXPECT_EQ(rt.events()[3].node_index, 1);
}

TEST(RunNodes, RuntimeContextEventLogCapturesAllocatorMemory) {
  // With an allocator attached and event logging on, every recorded event
  // carries the allocator's live and peak memory, and the ``run_node`` events
  // summarise the per-node dispatch (duration + memory) through summary().
  using core::runtime::RuntimeEventAction;
  core::runtime::SimpleRawBufferAllocator alloc(/*capacity=*/16);
  RuntimeContext rt(
      KernelContext(DefaultOpset(18)),
      core::runtime::RuntimeContextOptions{.allocator = &alloc, .events_enabled = true});
  rt.Set("x", Tensor::FromFloat("x", {2}, {-1.0f, 2.0f}));
  rt.Set("z", Tensor::FromFloat("z", {2}, {10.0f, 20.0f}));
  rt.ClearEvents();

  std::vector<NodeProto> nodes;
  nodes.push_back(MakeNode("Abs", {"x"}, {"t"}));
  nodes.push_back(MakeNode("Add", {"t", "z"}, {"y"}));
  utils::RepeatedProtoField<NodeProto> node_field(nodes);
  core::runtime::ExecutionPlan plan(node_field, {});
  core::runtime::RuntimeSession session(plan);
  session.Run(rt);

  ASSERT_FALSE(rt.events().empty());
  // The two inputs already consume two allocator buffers, so every event
  // recorded while running the graph reports a non-zero live footprint whose
  // peak never falls below the live value.
  bool saw_run_node = false;
  for (const auto &ev : rt.events()) {
    EXPECT_GT(ev.allocated_bytes, 0);
    EXPECT_GE(ev.peak_bytes, ev.allocated_bytes);
    const std::string text = ev.summary();
    EXPECT_NE(text.find("mem="), std::string::npos);
    EXPECT_NE(text.find("peak="), std::string::npos);
    if (ev.action == RuntimeEventAction::kRunNode) {
      saw_run_node = true;
      EXPECT_NE(text.find("took"), std::string::npos);
    }
  }
  EXPECT_TRUE(saw_run_node);
}

// ---------------------------------------------------------------------------
// Shape::product() tests
// ---------------------------------------------------------------------------
TEST(ShapeProduct, EmptyShapeIsScalarWithProduct1) {
  const core::runtime::Shape s{};
  EXPECT_EQ(s.product(), 1);
}

TEST(ShapeProduct, SingleDimensionEqualsItself) {
  const core::runtime::Shape s{7};
  EXPECT_EQ(s.product(), 7);
}

TEST(ShapeProduct, MultipleDimensionsMultipliedTogether) {
  const core::runtime::Shape s{2, 3, 4};
  EXPECT_EQ(s.product(), 24);
}

TEST(ShapeProduct, ZeroDimensionProducesZero) {
  const core::runtime::Shape s{4, 0, 3};
  EXPECT_EQ(s.product(), 0);
}

// ---------------------------------------------------------------------------
// TensorFromProto tests
// ---------------------------------------------------------------------------

TEST(TensorFromProto, FloatTypedField) {
  TensorProto tp;
  tp.set_name("w");
  tp.ref_dims().push_back(2);
  tp.set_data_type(TensorProto::DataType::FLOAT);
  tp.add_float_data(1.0f);
  tp.add_float_data(2.0f);

  Tensor t = TensorFromProto(tp);
  EXPECT_EQ(t.name, "w");
  EXPECT_EQ(t.shape, (std::vector<int64_t>{2}));
  EXPECT_EQ(t.data_type, static_cast<int32_t>(TensorProto::DataType::FLOAT));
  ASSERT_EQ(t.element_count(), 2);
  EXPECT_FLOAT_EQ(t.AsFloat()[0], 1.0f);
  EXPECT_FLOAT_EQ(t.AsFloat()[1], 2.0f);
}

TEST(TensorFromProto, FloatRawData) {
  TensorProto tp;
  tp.set_name("r");
  tp.ref_dims().push_back(3);
  tp.set_data_type(TensorProto::DataType::FLOAT);
  const float vals[] = {3.0f, 4.0f, 5.0f};
  const auto *raw_ptr = reinterpret_cast<const uint8_t *>(vals);
  tp.ref_raw_data() = std::vector<uint8_t>(raw_ptr, raw_ptr + sizeof(vals));

  Tensor t = TensorFromProto(tp);
  EXPECT_EQ(t.name, "r");
  ASSERT_EQ(t.element_count(), 3);
  EXPECT_FLOAT_EQ(t.AsFloat()[0], 3.0f);
  EXPECT_FLOAT_EQ(t.AsFloat()[1], 4.0f);
  EXPECT_FLOAT_EQ(t.AsFloat()[2], 5.0f);
}

TEST(TensorFromProto, Int64TypedField) {
  TensorProto tp;
  tp.set_name("i64");
  tp.ref_dims().push_back(2);
  tp.set_data_type(TensorProto::DataType::INT64);
  tp.add_int64_data(100);
  tp.add_int64_data(-200);

  Tensor t = TensorFromProto(tp);
  EXPECT_EQ(t.data_type, static_cast<int32_t>(TensorProto::DataType::INT64));
  ASSERT_EQ(t.element_count(), 2);
  EXPECT_EQ(t.AsInt64()[0], 100);
  EXPECT_EQ(t.AsInt64()[1], -200);
}

TEST(TensorFromProto, Int32TypedField) {
  TensorProto tp;
  tp.set_name("i32");
  tp.ref_dims().push_back(2);
  tp.set_data_type(TensorProto::DataType::INT32);
  tp.add_int32_data(7);
  tp.add_int32_data(-3);

  Tensor t = TensorFromProto(tp);
  EXPECT_EQ(t.data_type, static_cast<int32_t>(TensorProto::DataType::INT32));
  ASSERT_EQ(t.element_count(), 2);
  EXPECT_EQ(t.AsInt32()[0], 7);
  EXPECT_EQ(t.AsInt32()[1], -3);
}

TEST(TensorFromProto, ScalarNoShape) {
  // Scalar TensorProto has no dims entry.
  TensorProto tp;
  tp.set_name("s");
  tp.set_data_type(TensorProto::DataType::FLOAT);
  tp.add_float_data(42.0f);

  Tensor t = TensorFromProto(tp);
  EXPECT_TRUE(t.shape.empty());
  EXPECT_EQ(t.element_count(), 1);
  EXPECT_FLOAT_EQ(t.AsFloat()[0], 42.0f);
}

// ---------------------------------------------------------------------------
// RunGraph tests
// ---------------------------------------------------------------------------

TEST(RunGraph, InitializersLoadedAndNodesRun) {
  // Graph:  out = Add(x_input, w_init)
  //   x_input is provided by the caller; w_init comes from the initializer.
  TensorProto init_tp;
  init_tp.set_name("w_init");
  init_tp.ref_dims().push_back(2);
  init_tp.set_data_type(TensorProto::DataType::FLOAT);
  init_tp.add_float_data(10.0f);
  init_tp.add_float_data(20.0f);

  GraphProto graph;
  *graph.add_initializer() = init_tp;
  NodeProto *node = graph.add_node();
  node->set_op_type("Add");
  node->add_input("x_input");
  node->add_input("w_init");
  node->add_output("out");

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("x_input", Tensor::FromFloat("x_input", {2}, {1.0f, 2.0f}));

  RunGraphViaSession(graph, rt);

  ASSERT_TRUE(rt.Has("out"));
  const float *res = rt.Get("out").AsFloat();
  EXPECT_FLOAT_EQ(res[0], 11.0f);
  EXPECT_FLOAT_EQ(res[1], 22.0f);
}

TEST(RunGraph, CallerInputOverridesInitializer) {
  // When the caller has already seeded a name that is also an initializer,
  // the caller's value must win.
  TensorProto init_tp;
  init_tp.set_name("w");
  init_tp.ref_dims().push_back(1);
  init_tp.set_data_type(TensorProto::DataType::FLOAT);
  init_tp.add_float_data(999.0f); // should be ignored

  GraphProto graph;
  *graph.add_initializer() = init_tp;
  NodeProto *node = graph.add_node();
  node->set_op_type("Abs");
  node->add_input("w");
  node->add_output("out");

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  // Override the initializer with the caller's value.
  rt.Set("w", Tensor::FromFloat("w", {1}, {-5.0f}));

  RunGraphViaSession(graph, rt);

  ASSERT_TRUE(rt.Has("out"));
  EXPECT_FLOAT_EQ(rt.Get("out").AsFloat()[0], 5.0f);
}

TEST(RunModel, DelayedInitializerLoadsFromFileAtRuntime) {
  namespace fs = std::filesystem;

  const fs::path weights_path =
      fs::temp_directory_path() / "onnx_light_delayed_initializer_file.bin";
  fs::remove(weights_path);
  {
    std::ofstream out(weights_path, std::ios::binary);
    ASSERT_TRUE(out.good());
    const char pad[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    out.write(pad, sizeof(pad));
    const float values[2] = {1.5f, -2.0f};
    out.write(reinterpret_cast<const char *>(values), sizeof(values));
  }

  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *rt_os = model.add_opset_import();
  rt_os->set_domain("ai.rt");
  rt_os->set_version(1);
  GraphProto *graph = model.add_graph();
  graph->set_name("main");
  graph->add_output()->set_name("Y");
  NodeProto *node = graph->add_node();
  node->set_op_type("DelayedInitializer");
  node->set_domain("ai.rt");
  node->add_output("Y");

  AttributeProto *shape = node->add_attribute();
  shape->set_name("shape");
  shape->set_type(AttributeProto::AttributeType::INTS);
  shape->add_ints(2);

  AttributeProto *dtype = node->add_attribute();
  dtype->set_name("dtype");
  dtype->set_type(AttributeProto::AttributeType::INT);
  dtype->set_i(static_cast<int64_t>(TensorProto::DataType::FLOAT));

  AttributeProto *load_device = node->add_attribute();
  load_device->set_name("load_device");
  load_device->set_type(AttributeProto::AttributeType::STRING);
  load_device->set_s("file");

  AttributeProto *runtime_device = node->add_attribute();
  runtime_device->set_name("runtime_device");
  runtime_device->set_type(AttributeProto::AttributeType::STRING);
  runtime_device->set_s("cpu");

  AttributeProto *filename = node->add_attribute();
  filename->set_name("filename");
  filename->set_type(AttributeProto::AttributeType::STRING);
  filename->set_s(weights_path.string());

  AttributeProto *offset = node->add_attribute();
  offset->set_name("offset");
  offset->set_type(AttributeProto::AttributeType::INT);
  offset->set_i(8);

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  RunModelViaSession(model, rt);

  ASSERT_TRUE(rt.Has("Y"));
  const Tensor &y = rt.Get("Y");
  EXPECT_EQ(y.data_type, static_cast<int32_t>(TensorProto::DataType::FLOAT));
  ASSERT_EQ(y.shape.size(), 1u);
  EXPECT_EQ(y.shape[0], 2);
  EXPECT_FLOAT_EQ(y.AsFloat()[0], 1.5f);
  EXPECT_FLOAT_EQ(y.AsFloat()[1], -2.0f);
  fs::remove(weights_path);
}

TEST(RunModel, DelayedInitializerLoadsIntoCpuAtKernelInitialization) {
  namespace fs = std::filesystem;

  const fs::path weights_path =
      fs::temp_directory_path() / "onnx_light_delayed_initializer_cpu.bin";
  fs::remove(weights_path);
  {
    std::ofstream out(weights_path, std::ios::binary);
    ASSERT_TRUE(out.good());
    const float values[3] = {3.0f, 4.0f, 5.0f};
    out.write(reinterpret_cast<const char *>(values), sizeof(values));
  }

  NodeProto node = MakeNode("DelayedInitializer", {}, {"Y"}, "ai.rt");

  AttributeProto *shape = node.add_attribute();
  shape->set_name("shape");
  shape->set_type(AttributeProto::AttributeType::INTS);
  shape->add_ints(3);

  AttributeProto *dtype = node.add_attribute();
  dtype->set_name("dtype");
  dtype->set_type(AttributeProto::AttributeType::INT);
  dtype->set_i(static_cast<int64_t>(TensorProto::DataType::FLOAT));

  AttributeProto *load_device = node.add_attribute();
  load_device->set_name("load_device");
  load_device->set_type(AttributeProto::AttributeType::STRING);
  load_device->set_s("cpu");

  AttributeProto *runtime_device = node.add_attribute();
  runtime_device->set_name("runtime_device");
  runtime_device->set_type(AttributeProto::AttributeType::STRING);
  runtime_device->set_s("cpu");

  AttributeProto *filename = node.add_attribute();
  filename->set_name("filename");
  filename->set_type(AttributeProto::AttributeType::STRING);
  filename->set_s(weights_path.string());

  AttributeProto *offset = node.add_attribute();
  offset->set_name("offset");
  offset->set_type(AttributeProto::AttributeType::INT);
  offset->set_i(0);

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  RunNode(node, rt);

  ASSERT_TRUE(rt.Has("Y"));
  const Tensor &y = rt.Get("Y");
  EXPECT_EQ(y.data_type, static_cast<int32_t>(TensorProto::DataType::FLOAT));
  ASSERT_EQ(y.shape.size(), 1u);
  EXPECT_EQ(y.shape[0], 3);
  EXPECT_FLOAT_EQ(y.AsFloat()[0], 3.0f);
  EXPECT_FLOAT_EQ(y.AsFloat()[1], 4.0f);
  EXPECT_FLOAT_EQ(y.AsFloat()[2], 5.0f);
  fs::remove(weights_path);
}

TEST(RunModel, DelayedInitializerUsesAllocatorWhenProvided) {
  namespace fs = std::filesystem;

  const fs::path weights_path =
      fs::temp_directory_path() / "onnx_light_delayed_initializer_allocator.bin";
  fs::remove(weights_path);
  {
    std::ofstream out(weights_path, std::ios::binary);
    ASSERT_TRUE(out.good());
    const float values[2] = {10.0f, -3.5f};
    out.write(reinterpret_cast<const char *>(values), sizeof(values));
  }

  onnx_kernels::kernel::DelayedInitializer::Attributes attrs;
  attrs.shape = {2};
  attrs.dtype = static_cast<int32_t>(TensorProto::DataType::FLOAT);
  attrs.load_device = "file";
  attrs.runtime_device = "cpu";
  attrs.filename = weights_path.string();
  attrs.offset = 0;
  onnx_kernels::kernel::DelayedInitializer delayed(KernelContext(DefaultOpset(18)),
                                                   std::move(attrs));

  // SimpleRawBufferAllocator capacity counts buffer slots, not bytes.
  constexpr size_t kAllocatorSlotCapacity = 1;
  core::runtime::SimpleRawBufferAllocator alloc(kAllocatorSlotCapacity);
  RuntimeContext rt(KernelContext(DefaultOpset(18)),
                    core::runtime::RuntimeContextOptions{.allocator = &alloc});

  Tensor y = delayed(&rt);
  ASSERT_TRUE(y.has_allocation());
  EXPECT_EQ(y.allocation_owner(), &alloc);
  ASSERT_EQ(y.shape.size(), 1u);
  EXPECT_EQ(y.shape[0], 2);
  EXPECT_FLOAT_EQ(y.AsFloat()[0], 10.0f);
  EXPECT_FLOAT_EQ(y.AsFloat()[1], -3.5f);

  fs::remove(weights_path);
}

TEST(RunModel, DelayedInitializerCpuLoadUsesConstructionAllocator) {
  namespace fs = std::filesystem;

  const fs::path weights_path =
      fs::temp_directory_path() / "onnx_light_delayed_initializer_cpu_alloc.bin";
  fs::remove(weights_path);
  {
    std::ofstream out(weights_path, std::ios::binary);
    ASSERT_TRUE(out.good());
    const float values[3] = {1.0f, 2.0f, 3.0f};
    out.write(reinterpret_cast<const char *>(values), sizeof(values));
  }

  // SimpleRawBufferAllocator capacity counts buffer slots, not bytes. One slot
  // is consumed by loaded_bytes_ at construction, and a second by the output
  // produced by operator() through the construction-time allocator.
  constexpr size_t kAllocatorSlotCapacity = 2;
  core::runtime::SimpleRawBufferAllocator alloc(kAllocatorSlotCapacity);

  KernelContext ctx(DefaultOpset(18));
  ctx.allocator = &alloc;

  onnx_kernels::kernel::DelayedInitializer::Attributes attrs;
  attrs.shape = {3};
  attrs.dtype = static_cast<int32_t>(TensorProto::DataType::FLOAT);
  attrs.load_device = "cpu";
  attrs.runtime_device = "cpu";
  attrs.filename = weights_path.string();
  attrs.offset = 0;
  // Construction with an allocator: loaded_bytes_ should be allocator-backed.
  onnx_kernels::kernel::DelayedInitializer delayed(ctx, std::move(attrs));

  // The internal buffer consumed one allocator slot at construction time.
  EXPECT_EQ(alloc.TotalAllocatedSize(), 3 * sizeof(float));

  // Calling operator() without a runtime allocator falls back to the
  // construction-time allocator, so the output is allocator-backed.
  Tensor y = delayed(nullptr);
  ASSERT_TRUE(y.has_allocation());
  EXPECT_EQ(y.allocation_owner(), &alloc);
  ASSERT_EQ(y.shape.size(), 1u);
  EXPECT_EQ(y.shape[0], 3);
  EXPECT_FLOAT_EQ(y.AsFloat()[0], 1.0f);
  EXPECT_FLOAT_EQ(y.AsFloat()[1], 2.0f);
  EXPECT_FLOAT_EQ(y.AsFloat()[2], 3.0f);

  fs::remove(weights_path);
}

TEST(RunModel, DelayedInitializerCpuLoadWithBothAllocators) {
  namespace fs = std::filesystem;

  const fs::path weights_path =
      fs::temp_directory_path() / "onnx_light_delayed_initializer_cpu_both_alloc.bin";
  fs::remove(weights_path);
  {
    std::ofstream out(weights_path, std::ios::binary);
    ASSERT_TRUE(out.good());
    const float values[2] = {7.0f, -8.5f};
    out.write(reinterpret_cast<const char *>(values), sizeof(values));
  }

  // Allocator for the construction-time loaded_bytes_ buffer.
  // SimpleRawBufferAllocator capacity counts buffer slots, not bytes.
  constexpr size_t kConstructionAllocatorSlots = 1;
  core::runtime::SimpleRawBufferAllocator construction_alloc(kConstructionAllocatorSlots);

  KernelContext ctx(DefaultOpset(18));
  ctx.allocator = &construction_alloc;

  onnx_kernels::kernel::DelayedInitializer::Attributes attrs;
  attrs.shape = {2};
  attrs.dtype = static_cast<int32_t>(TensorProto::DataType::FLOAT);
  attrs.load_device = "cpu";
  attrs.runtime_device = "cpu";
  attrs.filename = weights_path.string();
  attrs.offset = 0;
  onnx_kernels::kernel::DelayedInitializer delayed(ctx, std::move(attrs));

  // Call operator() with a separate runtime allocator: the output must be
  // backed by the runtime allocator, not the construction-time allocator.
  constexpr size_t kRuntimeAllocatorSlots = 1;
  core::runtime::SimpleRawBufferAllocator runtime_alloc(kRuntimeAllocatorSlots);
  RuntimeContext rt(KernelContext(DefaultOpset(18)),
                    core::runtime::RuntimeContextOptions{.allocator = &runtime_alloc});

  Tensor y = delayed(&rt);
  ASSERT_TRUE(y.has_allocation());
  EXPECT_EQ(y.allocation_owner(), &runtime_alloc);
  ASSERT_EQ(y.shape.size(), 1u);
  EXPECT_EQ(y.shape[0], 2);
  EXPECT_FLOAT_EQ(y.AsFloat()[0], 7.0f);
  EXPECT_FLOAT_EQ(y.AsFloat()[1], -8.5f);

  fs::remove(weights_path);
}

// ---------------------------------------------------------------------------
// RunFunction tests
// ---------------------------------------------------------------------------

TEST(RunFunction, NodesRun) {
  FunctionProto func;
  func.set_name("f");
  func.add_input("a");
  func.add_input("b");
  func.add_output("result");
  NodeProto *node = func.add_node();
  node->set_op_type("Mul");
  node->add_input("a");
  node->add_input("b");
  node->add_output("result");

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("a", Tensor::FromFloat("a", {2}, {3.0f, 4.0f}));
  rt.Set("b", Tensor::FromFloat("b", {2}, {2.0f, 5.0f}));

  const ExecutionPlan &plan = rt.GetExecutionPlan(func);
  RuntimeSession session(plan);
  session.Run(rt);

  ASSERT_TRUE(rt.Has("result"));
  const float *res = rt.Get("result").AsFloat();
  EXPECT_FLOAT_EQ(res[0], 6.0f);
  EXPECT_FLOAT_EQ(res[1], 20.0f);
}

// ---------------------------------------------------------------------------
// RunModel tests
// ---------------------------------------------------------------------------

TEST(RunModel, GraphRun) {
  // Build a minimal ModelProto with a single Add node.
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(18);
  GraphProto *g = model.add_graph();
  g->set_name("test");
  NodeProto *node = g->add_node();
  node->set_op_type("Add");
  node->add_input("p");
  node->add_input("q");
  node->add_output("r");

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("p", Tensor::FromFloat("p", {3}, {1.0f, 2.0f, 3.0f}));
  rt.Set("q", Tensor::FromFloat("q", {3}, {4.0f, 5.0f, 6.0f}));

  RunModelViaSession(model, rt);

  ASSERT_TRUE(rt.Has("r"));
  const float *res = rt.Get("r").AsFloat();
  EXPECT_FLOAT_EQ(res[0], 5.0f);
  EXPECT_FLOAT_EQ(res[1], 7.0f);
  EXPECT_FLOAT_EQ(res[2], 9.0f);
}

TEST(RunModel, NoGraphThrows) {
  ModelProto model;
  model.set_ir_version(10);
  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  EXPECT_THROW(RunModelViaSession(model, rt), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Model-local function dispatch tests
// ---------------------------------------------------------------------------

TEST(RunModel, NodeDispatchedToModelLocalFunction) {
  // Define a model-local function "MyAddMul" in domain "custom":
  //   inputs:  a, b, c
  //   output:  out  = (a + b) * c
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(18);
  OperatorSetIdProto *custom_os = model.add_opset_import();
  custom_os->set_domain("custom");
  custom_os->set_version(1);

  FunctionProto *func = model.add_functions();
  func->set_name("MyAddMul");
  func->set_domain("custom");
  func->add_input("a");
  func->add_input("b");
  func->add_input("c");
  func->add_output("out");
  {
    NodeProto *n = func->add_node();
    n->set_op_type("Add");
    n->add_input("a");
    n->add_input("b");
    n->add_output("tmp");
  }
  {
    NodeProto *n = func->add_node();
    n->set_op_type("Mul");
    n->add_input("tmp");
    n->add_input("c");
    n->add_output("out");
  }

  // Main graph: y = MyAddMul(x, w, k) where (x, w, k) are graph inputs.
  GraphProto *g = model.add_graph();
  g->set_name("test");
  NodeProto *node = g->add_node();
  node->set_op_type("MyAddMul");
  node->set_domain("custom");
  node->add_input("x");
  node->add_input("w");
  node->add_input("k");
  node->add_output("y");

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("x", Tensor::FromFloat("x", {2}, {1.0f, 2.0f}));
  rt.Set("w", Tensor::FromFloat("w", {2}, {3.0f, 4.0f}));
  rt.Set("k", Tensor::FromFloat("k", {2}, {10.0f, 100.0f}));

  RunModelViaSession(model, rt);

  ASSERT_TRUE(rt.Has("y"));
  const float *res = rt.Get("y").AsFloat();
  ASSERT_EQ(rt.Get("y").element_count(), 2);
  EXPECT_FLOAT_EQ(res[0], (1.0f + 3.0f) * 10.0f);
  EXPECT_FLOAT_EQ(res[1], (2.0f + 4.0f) * 100.0f);
  // The function's internal "tmp" value must NOT leak into the caller's
  // tensor map: the child context is isolated.
  EXPECT_FALSE(rt.Has("tmp"));
}

TEST(RunModel, ModelLocalFunctionCanCallAnotherFunction) {
  // Define two model-local functions in "custom":
  //   AddOne(x) -> Add(x, one_init_passed_as_input)  -- here we use x+x
  //   Quad(x)   -> AddOne(AddOne(x))                 -- so result = x*4? No: just chained calls
  // To keep it simple and only exercise nesting, use:
  //   Twice(x)  -> Add(x, x)
  //   Quad(x)   -> Twice(Twice(x))   => 4*x
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(18);

  FunctionProto *twice = model.add_functions();
  twice->set_name("Twice");
  twice->set_domain("custom");
  twice->add_input("x");
  twice->add_output("y");
  {
    NodeProto *n = twice->add_node();
    n->set_op_type("Add");
    n->add_input("x");
    n->add_input("x");
    n->add_output("y");
  }

  FunctionProto *quad = model.add_functions();
  quad->set_name("Quad");
  quad->set_domain("custom");
  quad->add_input("x");
  quad->add_output("y");
  {
    NodeProto *n = quad->add_node();
    n->set_op_type("Twice");
    n->set_domain("custom");
    n->add_input("x");
    n->add_output("t");
  }
  {
    NodeProto *n = quad->add_node();
    n->set_op_type("Twice");
    n->set_domain("custom");
    n->add_input("t");
    n->add_output("y");
  }

  GraphProto *g = model.add_graph();
  g->set_name("test");
  NodeProto *node = g->add_node();
  node->set_op_type("Quad");
  node->set_domain("custom");
  node->add_input("x");
  node->add_output("y");

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("x", Tensor::FromFloat("x", {3}, {1.0f, 2.5f, -3.0f}));

  RunModelViaSession(model, rt);

  ASSERT_TRUE(rt.Has("y"));
  const float *res = rt.Get("y").AsFloat();
  ASSERT_EQ(rt.Get("y").element_count(), 3);
  EXPECT_FLOAT_EQ(res[0], 4.0f);
  EXPECT_FLOAT_EQ(res[1], 10.0f);
  EXPECT_FLOAT_EQ(res[2], -12.0f);
}

TEST(RunModel, ModelLocalFunctionCallsAnotherFunctionAcrossDomains) {
  // Define two model-local functions in different domains; the function
  // in domain "outer" calls into the function in domain "inner". This
  // exercises that the function registry propagated to the child
  // RuntimeContext is keyed by (domain, name, overload) and that
  // cross-domain function-to-function dispatch works.
  //
  //   inner::Square(x) -> Mul(x, x)
  //   outer::SquareThenAdd(x, y) -> Add(inner::Square(x), y)
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(18);

  FunctionProto *square = model.add_functions();
  square->set_name("Square");
  square->set_domain("inner");
  square->add_input("x");
  square->add_output("y");
  {
    NodeProto *n = square->add_node();
    n->set_op_type("Mul");
    n->add_input("x");
    n->add_input("x");
    n->add_output("y");
  }

  FunctionProto *sqadd = model.add_functions();
  sqadd->set_name("SquareThenAdd");
  sqadd->set_domain("outer");
  sqadd->add_input("a");
  sqadd->add_input("b");
  sqadd->add_output("z");
  {
    NodeProto *n = sqadd->add_node();
    n->set_op_type("Square");
    n->set_domain("inner");
    n->add_input("a");
    n->add_output("a2");
  }
  {
    NodeProto *n = sqadd->add_node();
    n->set_op_type("Add");
    n->add_input("a2");
    n->add_input("b");
    n->add_output("z");
  }

  GraphProto *g = model.add_graph();
  g->set_name("test");
  NodeProto *node = g->add_node();
  node->set_op_type("SquareThenAdd");
  node->set_domain("outer");
  node->add_input("x");
  node->add_input("y");
  node->add_output("z");

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("x", Tensor::FromFloat("x", {3}, {1.0f, 2.0f, 3.0f}));
  rt.Set("y", Tensor::FromFloat("y", {3}, {10.0f, 20.0f, 30.0f}));

  RunModelViaSession(model, rt);

  ASSERT_TRUE(rt.Has("z"));
  const float *res = rt.Get("z").AsFloat();
  ASSERT_EQ(rt.Get("z").element_count(), 3);
  EXPECT_FLOAT_EQ(res[0], 1.0f * 1.0f + 10.0f);
  EXPECT_FLOAT_EQ(res[1], 2.0f * 2.0f + 20.0f);
  EXPECT_FLOAT_EQ(res[2], 3.0f * 3.0f + 30.0f);
  // The intermediate name produced inside the SquareThenAdd function
  // body must not leak into the caller's tensor map.
  EXPECT_FALSE(rt.Has("a2"));
}

TEST(RunModel, ModelLocalFunctionThreeLevelNestedCalls) {
  // Demonstrates that the function registry is propagated through
  // arbitrary nesting depth: the top-level graph calls Outer, Outer
  // calls Middle, and Middle calls Inner. Each level is a separate
  // FunctionProto in the model.
  //
  //   Inner(x)  -> Add(x, x)        => 2*x
  //   Middle(x) -> Inner(Inner(x))  => 4*x
  //   Outer(x)  -> Middle(Inner(x)) => 8*x
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(18);

  FunctionProto *inner = model.add_functions();
  inner->set_name("Inner");
  inner->set_domain("custom");
  inner->add_input("x");
  inner->add_output("y");
  {
    NodeProto *n = inner->add_node();
    n->set_op_type("Add");
    n->add_input("x");
    n->add_input("x");
    n->add_output("y");
  }

  FunctionProto *middle = model.add_functions();
  middle->set_name("Middle");
  middle->set_domain("custom");
  middle->add_input("x");
  middle->add_output("y");
  {
    NodeProto *n = middle->add_node();
    n->set_op_type("Inner");
    n->set_domain("custom");
    n->add_input("x");
    n->add_output("t");
  }
  {
    NodeProto *n = middle->add_node();
    n->set_op_type("Inner");
    n->set_domain("custom");
    n->add_input("t");
    n->add_output("y");
  }

  FunctionProto *outer = model.add_functions();
  outer->set_name("Outer");
  outer->set_domain("custom");
  outer->add_input("x");
  outer->add_output("y");
  {
    NodeProto *n = outer->add_node();
    n->set_op_type("Inner");
    n->set_domain("custom");
    n->add_input("x");
    n->add_output("t");
  }
  {
    NodeProto *n = outer->add_node();
    n->set_op_type("Middle");
    n->set_domain("custom");
    n->add_input("t");
    n->add_output("y");
  }

  GraphProto *g = model.add_graph();
  g->set_name("test");
  NodeProto *node = g->add_node();
  node->set_op_type("Outer");
  node->set_domain("custom");
  node->add_input("x");
  node->add_output("y");

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("x", Tensor::FromFloat("x", {3}, {1.0f, 2.5f, -3.0f}));

  RunModelViaSession(model, rt);

  ASSERT_TRUE(rt.Has("y"));
  const float *res = rt.Get("y").AsFloat();
  ASSERT_EQ(rt.Get("y").element_count(), 3);
  EXPECT_FLOAT_EQ(res[0], 8.0f);
  EXPECT_FLOAT_EQ(res[1], 20.0f);
  EXPECT_FLOAT_EQ(res[2], -24.0f);
}

TEST(RunModel, ModelLocalFunctionOverloadDisambiguation) {
  // Two functions share the same (domain, name) but differ by overload.
  // The dispatcher must pick the one matching node.overload.
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(18);

  FunctionProto *f_sum = model.add_functions();
  f_sum->set_name("Combine");
  f_sum->set_domain("custom");
  f_sum->set_overload("sum");
  f_sum->add_input("a");
  f_sum->add_input("b");
  f_sum->add_output("out");
  {
    NodeProto *n = f_sum->add_node();
    n->set_op_type("Add");
    n->add_input("a");
    n->add_input("b");
    n->add_output("out");
  }

  FunctionProto *f_diff = model.add_functions();
  f_diff->set_name("Combine");
  f_diff->set_domain("custom");
  f_diff->set_overload("diff");
  f_diff->add_input("a");
  f_diff->add_input("b");
  f_diff->add_output("out");
  {
    NodeProto *n = f_diff->add_node();
    n->set_op_type("Sub");
    n->add_input("a");
    n->add_input("b");
    n->add_output("out");
  }

  GraphProto *g = model.add_graph();
  g->set_name("test");
  NodeProto *n1 = g->add_node();
  n1->set_op_type("Combine");
  n1->set_domain("custom");
  n1->set_overload("sum");
  n1->add_input("x");
  n1->add_input("y");
  n1->add_output("s");
  NodeProto *n2 = g->add_node();
  n2->set_op_type("Combine");
  n2->set_domain("custom");
  n2->set_overload("diff");
  n2->add_input("x");
  n2->add_input("y");
  n2->add_output("d");

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("x", Tensor::FromFloat("x", {1}, {10.0f}));
  rt.Set("y", Tensor::FromFloat("y", {1}, {3.0f}));

  RunModelViaSession(model, rt);

  ASSERT_TRUE(rt.Has("s"));
  ASSERT_TRUE(rt.Has("d"));
  EXPECT_FLOAT_EQ(rt.Get("s").AsFloat()[0], 13.0f);
  EXPECT_FLOAT_EQ(rt.Get("d").AsFloat()[0], 7.0f);
}

TEST(RunModel, ModelLocalFunctionWrongInputCountThrows) {
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(18);

  FunctionProto *func = model.add_functions();
  func->set_name("F");
  func->set_domain("custom");
  func->add_input("a");
  func->add_input("b");
  func->add_output("out");
  NodeProto *fn = func->add_node();
  fn->set_op_type("Add");
  fn->add_input("a");
  fn->add_input("b");
  fn->add_output("out");

  GraphProto *g = model.add_graph();
  NodeProto *node = g->add_node();
  node->set_op_type("F");
  node->set_domain("custom");
  node->add_input("x"); // only 1, function expects 2
  node->add_output("y");

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("x", Tensor::FromFloat("x", {1}, {1.0f}));

  EXPECT_THROW(RunModelViaSession(model, rt), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Model-local function with linked (ref_attr_name) attributes
// ---------------------------------------------------------------------------

namespace {

// Builds a constant-only sub-graph that emits a single FLOAT scalar
// named ``out_name`` set to ``value``. Used as a GRAPH attribute below.
void FillConstantBranch(GraphProto &g, const std::string &branch_name, const std::string &init_name,
                        const std::string &out_name, float value) {
  g.set_name(branch_name);
  TensorProto *init = g.add_initializer();
  init->set_name(init_name);
  init->set_data_type(TensorProto::DataType::FLOAT);
  init->add_float_data(value);
  // The If implementation expects each branch sub-graph to produce its
  // output via at least one node. Use Add(init, init) so the value is
  // doubled, mirroring the existing ``IfNodeWithBranchSubgraphs`` test.
  NodeProto *add = g.add_node();
  add->set_op_type("Add");
  add->add_input(init_name);
  add->add_input(init_name);
  add->add_output(out_name);
  g.add_output()->set_name(out_name);
}

} // namespace

TEST(RunModel, ModelLocalFunctionLinkedAttributeFromCallSite) {
  // Define a model-local function "Pick(cond)" whose body delegates to
  // an ``If`` node where both ``then_branch`` and ``else_branch`` are
  // attribute references (``ref_attr_name``) to call-site attributes of
  // the same name. Two distinct call-sites supply different branch
  // sub-graphs, proving that the attribute is resolved per call rather
  // than baked in at function-definition time.
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(18);
  OperatorSetIdProto *custom_os = model.add_opset_import();
  custom_os->set_domain("custom");
  custom_os->set_version(1);

  FunctionProto *func = model.add_functions();
  func->set_name("Pick");
  func->set_domain("custom");
  func->add_input("cond");
  func->add_output("out");
  func->add_attribute("then_branch");
  func->add_attribute("else_branch");
  {
    NodeProto *n = func->add_node();
    n->set_op_type("If");
    n->add_input("cond");
    n->add_output("out");
    AttributeProto *tref = n->add_attribute();
    tref->set_name("then_branch");
    tref->set_ref_attr_name("then_branch");
    tref->set_type(AttributeProto::AttributeType::GRAPH);
    AttributeProto *eref = n->add_attribute();
    eref->set_name("else_branch");
    eref->set_ref_attr_name("else_branch");
    eref->set_type(AttributeProto::AttributeType::GRAPH);
  }

  GraphProto *g = model.add_graph();
  g->set_name("test");
  NodeProto *call = g->add_node();
  call->set_op_type("Pick");
  call->set_domain("custom");
  call->add_input("cond");
  call->add_output("out");
  AttributeProto *tattr = call->add_attribute();
  tattr->set_name("then_branch");
  tattr->set_type(AttributeProto::AttributeType::GRAPH);
  FillConstantBranch(*tattr->add_g(), "then_g", "t", "z", 10.0f);
  AttributeProto *eattr = call->add_attribute();
  eattr->set_name("else_branch");
  eattr->set_type(AttributeProto::AttributeType::GRAPH);
  FillConstantBranch(*eattr->add_g(), "else_g", "e", "z", 1.0f);

  RuntimeContext rt_true(KernelContext(DefaultOpset(18)));
  rt_true.Set("cond", Tensor::FromBool("cond", {}, {1}));
  RunModelViaSession(model, rt_true);
  ASSERT_TRUE(rt_true.Has("out"));
  EXPECT_FLOAT_EQ(rt_true.Get("out").AsFloat()[0], 20.0f);

  RuntimeContext rt_false(KernelContext(DefaultOpset(18)));
  rt_false.Set("cond", Tensor::FromBool("cond", {}, {0}));
  RunModelViaSession(model, rt_false);
  ASSERT_TRUE(rt_false.Has("out"));
  EXPECT_FLOAT_EQ(rt_false.Get("out").AsFloat()[0], 2.0f);
}

TEST(RunModel, ModelLocalFunctionLinkedAttributeUsesDefault) {
  // The function declares typed defaults for ``then_branch`` and
  // ``else_branch`` via ``attribute_proto``. The call-site omits both
  // and the defaults must be used instead.
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(18);
  OperatorSetIdProto *custom_os = model.add_opset_import();
  custom_os->set_domain("custom");
  custom_os->set_version(1);

  FunctionProto *func = model.add_functions();
  func->set_name("Pick");
  func->set_domain("custom");
  func->add_input("cond");
  func->add_output("out");
  AttributeProto *tdef = func->add_attribute_proto();
  tdef->set_name("then_branch");
  tdef->set_type(AttributeProto::AttributeType::GRAPH);
  FillConstantBranch(*tdef->add_g(), "then_default_g", "t_def", "z", 5.0f);
  AttributeProto *edef = func->add_attribute_proto();
  edef->set_name("else_branch");
  edef->set_type(AttributeProto::AttributeType::GRAPH);
  FillConstantBranch(*edef->add_g(), "else_default_g", "e_def", "z", 7.0f);
  {
    NodeProto *n = func->add_node();
    n->set_op_type("If");
    n->add_input("cond");
    n->add_output("out");
    AttributeProto *tref = n->add_attribute();
    tref->set_name("then_branch");
    tref->set_ref_attr_name("then_branch");
    tref->set_type(AttributeProto::AttributeType::GRAPH);
    AttributeProto *eref = n->add_attribute();
    eref->set_name("else_branch");
    eref->set_ref_attr_name("else_branch");
    eref->set_type(AttributeProto::AttributeType::GRAPH);
  }

  GraphProto *g = model.add_graph();
  g->set_name("test");
  NodeProto *call = g->add_node();
  call->set_op_type("Pick");
  call->set_domain("custom");
  call->add_input("cond");
  call->add_output("out");

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("cond", Tensor::FromBool("cond", {}, {1}));
  RunModelViaSession(model, rt);
  ASSERT_TRUE(rt.Has("out"));
  EXPECT_FLOAT_EQ(rt.Get("out").AsFloat()[0], 10.0f);
}

TEST(RunModel, ModelLocalFunctionDoesNotMutateModel) {
  // Verify that resolving ``ref_attr_name`` references operates on a
  // copy of the FunctionProto so the source ModelProto's function body
  // is unchanged after running the model.
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(18);
  OperatorSetIdProto *custom_os = model.add_opset_import();
  custom_os->set_domain("custom");
  custom_os->set_version(1);

  FunctionProto *func = model.add_functions();
  func->set_name("Pick");
  func->set_domain("custom");
  func->add_input("cond");
  func->add_output("out");
  func->add_attribute("then_branch");
  func->add_attribute("else_branch");
  {
    NodeProto *n = func->add_node();
    n->set_op_type("If");
    n->add_input("cond");
    n->add_output("out");
    AttributeProto *tref = n->add_attribute();
    tref->set_name("then_branch");
    tref->set_ref_attr_name("then_branch");
    tref->set_type(AttributeProto::AttributeType::GRAPH);
    AttributeProto *eref = n->add_attribute();
    eref->set_name("else_branch");
    eref->set_ref_attr_name("else_branch");
    eref->set_type(AttributeProto::AttributeType::GRAPH);
  }

  GraphProto *g = model.add_graph();
  g->set_name("test");
  NodeProto *call = g->add_node();
  call->set_op_type("Pick");
  call->set_domain("custom");
  call->add_input("cond");
  call->add_output("out");
  AttributeProto *tattr = call->add_attribute();
  tattr->set_name("then_branch");
  tattr->set_type(AttributeProto::AttributeType::GRAPH);
  FillConstantBranch(*tattr->add_g(), "then_g", "t", "z", 10.0f);
  AttributeProto *eattr = call->add_attribute();
  eattr->set_name("else_branch");
  eattr->set_type(AttributeProto::AttributeType::GRAPH);
  FillConstantBranch(*eattr->add_g(), "else_g", "e", "z", 1.0f);

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("cond", Tensor::FromBool("cond", {}, {1}));
  RunModelViaSession(model, rt);

  // The function body's attribute must still be a reference (no graph
  // baked in) so the same model can be executed again with a different
  // call-site attribute.
  const FunctionProto &saved = model.functions()[0];
  const AttributeProto &a = saved.node()[0].attribute()[0];
  EXPECT_EQ(a.ref_attr_name(), "then_branch");
  EXPECT_FALSE(a.has_g());
}

TEST(RunModel, IfNodeWithBranchSubgraphs) {
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(18);

  GraphProto *g = model.add_graph();
  g->set_name("main");
  NodeProto *if_node = g->add_node();
  if_node->set_op_type("If");
  if_node->add_input("cond");
  if_node->add_output("out");

  AttributeProto *then_attr = if_node->add_attribute();
  then_attr->set_name("then_branch");
  then_attr->set_type(AttributeProto::AttributeType::GRAPH);
  GraphProto *then_g = then_attr->add_g();
  then_g->set_name("then_graph");
  TensorProto *then_init = then_g->add_initializer();
  then_init->set_name("t");
  then_init->set_data_type(TensorProto::DataType::FLOAT);
  then_init->add_float_data(10.0f);
  NodeProto *then_add = then_g->add_node();
  then_add->set_op_type("Add");
  then_add->add_input("t");
  then_add->add_input("t");
  then_add->add_output("z");
  then_g->add_output()->set_name("z");

  AttributeProto *else_attr = if_node->add_attribute();
  else_attr->set_name("else_branch");
  else_attr->set_type(AttributeProto::AttributeType::GRAPH);
  GraphProto *else_g = else_attr->add_g();
  else_g->set_name("else_graph");
  TensorProto *else_init = else_g->add_initializer();
  else_init->set_name("e");
  else_init->set_data_type(TensorProto::DataType::FLOAT);
  else_init->add_float_data(1.0f);
  NodeProto *else_add = else_g->add_node();
  else_add->set_op_type("Add");
  else_add->add_input("e");
  else_add->add_input("e");
  else_add->add_output("z");
  else_g->add_output()->set_name("z");

  RuntimeContext rt_true(KernelContext(DefaultOpset(18)));
  rt_true.Set("cond", Tensor::FromBool("cond", {}, {1}));
  RunModelViaSession(model, rt_true);
  ASSERT_TRUE(rt_true.Has("out"));
  EXPECT_FLOAT_EQ(rt_true.Get("out").AsFloat()[0], 20.0f);

  RuntimeContext rt_false(KernelContext(DefaultOpset(18)));
  rt_false.Set("cond", Tensor::FromBool("cond", {}, {0}));
  RunModelViaSession(model, rt_false);
  ASSERT_TRUE(rt_false.Has("out"));
  EXPECT_FLOAT_EQ(rt_false.Get("out").AsFloat()[0], 2.0f);
}

TEST(RunModel, LoopNodeRunsBodySubgraph) {
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(18);

  GraphProto *g = model.add_graph();
  g->set_name("main");
  NodeProto *loop = g->add_node();
  loop->set_op_type("Loop");
  loop->add_input("M");
  loop->add_input("cond");
  loop->add_input("s_init");
  loop->add_output("s_final");
  loop->add_output("scan");

  AttributeProto *body_attr = loop->add_attribute();
  body_attr->set_name("body");
  body_attr->set_type(AttributeProto::AttributeType::GRAPH);
  GraphProto *body = body_attr->add_g();
  body->set_name("loop_body");
  body->add_input()->set_name("iter");
  body->add_input()->set_name("cond_in");
  body->add_input()->set_name("s_in");
  TensorProto *one = body->add_initializer();
  one->set_name("one");
  one->set_data_type(TensorProto::DataType::FLOAT);
  one->add_float_data(1.0f);
  NodeProto *add = body->add_node();
  add->set_op_type("Add");
  add->add_input("s_in");
  add->add_input("one");
  add->add_output("s_out");
  body->add_output()->set_name("cond_in");
  body->add_output()->set_name("s_out");
  body->add_output()->set_name("s_out");

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("M", Tensor::FromInt64("M", {}, {3}));
  rt.Set("cond", Tensor::FromBool("cond", {}, {1}));
  rt.Set("s_init", Tensor::FromFloat("s_init", {}, {0.0f}));

  RunModelViaSession(model, rt);

  ASSERT_TRUE(rt.Has("s_final"));
  ASSERT_TRUE(rt.Has("scan"));
  EXPECT_FLOAT_EQ(rt.Get("s_final").AsFloat()[0], 3.0f);
  ASSERT_EQ(rt.Get("scan").shape, (std::vector<int64_t>{3}));
  const float *scan = rt.Get("scan").AsFloat();
  EXPECT_FLOAT_EQ(scan[0], 1.0f);
  EXPECT_FLOAT_EQ(scan[1], 2.0f);
  EXPECT_FLOAT_EQ(scan[2], 3.0f);
}

// Variant of LoopNodeRunsBodySubgraph with a SimpleRawBufferAllocator. Verifies
// that subgraph contexts created inside SubgraphSession::Run do not inherit the allocator,
// preventing double-free of body-output tensors threaded as loop-carried state
// across iterations. Iter/cond scalars are now allocated transiently via the
// parent allocator, but those slots are released before the final loop outputs
// are materialized, so the peak slot usage remains two.
TEST(RunModel, LoopNodeRunsBodySubgraphWithAllocator) {
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(18);

  GraphProto *g = model.add_graph();
  g->set_name("main");
  NodeProto *loop = g->add_node();
  loop->set_op_type("Loop");
  loop->add_input("M");
  loop->add_input("cond");
  loop->add_input("s_init");
  loop->add_output("s_final");
  loop->add_output("scan");

  AttributeProto *body_attr = loop->add_attribute();
  body_attr->set_name("body");
  body_attr->set_type(AttributeProto::AttributeType::GRAPH);
  GraphProto *body = body_attr->add_g();
  body->set_name("loop_body");
  body->add_input()->set_name("iter");
  body->add_input()->set_name("cond_in");
  body->add_input()->set_name("s_in");
  TensorProto *one = body->add_initializer();
  one->set_name("one");
  one->set_data_type(TensorProto::DataType::FLOAT);
  one->add_float_data(1.0f);
  NodeProto *add = body->add_node();
  add->set_op_type("Add");
  add->add_input("s_in");
  add->add_input("one");
  add->add_output("s_out");
  body->add_output()->set_name("cond_in");
  body->add_output()->set_name("s_out");
  body->add_output()->set_name("s_out");

  // Two slots are sufficient: the transient iter/cond scalar allocations are
  // freed at the end of each iteration before the final outputs are stored.
  constexpr size_t kAllocatorSlotCapacity = 2;
  core::runtime::SimpleRawBufferAllocator alloc(kAllocatorSlotCapacity);

  TensorMap tensors;
  tensors.emplace("M", Tensor::FromInt64("M", {}, {3}));
  tensors.emplace("cond", Tensor::FromBool("cond", {}, {1}));
  tensors.emplace("s_init", Tensor::FromFloat("s_init", {}, {0.0f}));
  RuntimeContext rt(KernelContext(DefaultOpset(18)), std::move(tensors),
                    core::runtime::RuntimeContextOptions{.allocator = &alloc});

  RunModelViaSession(model, rt);

  ASSERT_TRUE(rt.Has("s_final"));
  ASSERT_TRUE(rt.Has("scan"));
  EXPECT_FLOAT_EQ(rt.Get("s_final").AsFloat()[0], 3.0f);
  ASSERT_EQ(rt.Get("scan").shape, (std::vector<int64_t>{3}));
  const float *scan = rt.Get("scan").AsFloat();
  EXPECT_FLOAT_EQ(scan[0], 1.0f);
  EXPECT_FLOAT_EQ(scan[1], 2.0f);
  EXPECT_FLOAT_EQ(scan[2], 3.0f);
}

TEST(RunModel, LoopNodeAllocatorBacksTransientIterAndCondScalars) {
  CountingAllocator alloc(/*capacity=*/2);
  {
    ModelProto model;
    model.set_ir_version(10);
    OperatorSetIdProto *os = model.add_opset_import();
    os->set_version(18);

    GraphProto *g = model.add_graph();
    g->set_name("main");
    NodeProto *loop = g->add_node();
    loop->set_op_type("Loop");
    loop->add_input("M");
    loop->add_input("cond");
    loop->add_input("s_init");
    loop->add_output("s_final");
    loop->add_output("scan");

    AttributeProto *body_attr = loop->add_attribute();
    body_attr->set_name("body");
    body_attr->set_type(AttributeProto::AttributeType::GRAPH);
    GraphProto *body = body_attr->add_g();
    body->set_name("loop_body");
    body->add_input()->set_name("iter");
    body->add_input()->set_name("cond_in");
    body->add_input()->set_name("s_in");
    TensorProto *one = body->add_initializer();
    one->set_name("one");
    one->set_data_type(TensorProto::DataType::FLOAT);
    one->add_float_data(1.0f);
    NodeProto *add = body->add_node();
    add->set_op_type("Add");
    add->add_input("s_in");
    add->add_input("one");
    add->add_output("s_out");
    body->add_output()->set_name("cond_in");
    body->add_output()->set_name("s_out");
    body->add_output()->set_name("s_out");

    TensorMap tensors;
    tensors.emplace("M", Tensor::FromInt64("M", {}, {3}));
    tensors.emplace("cond", Tensor::FromBool("cond", {}, {1}));
    tensors.emplace("s_init", Tensor::FromFloat("s_init", {}, {0.0f}));
    RuntimeContext rt(KernelContext(DefaultOpset(18)), std::move(tensors),
                      core::runtime::RuntimeContextOptions{.allocator = &alloc});

    RunModelViaSession(model, rt);

    // 3 iterations x 2 transient scalar inputs (iter, cond_in) + 2 final loop
    // outputs (s_final, scan) = 8 allocations total. Only the 2 final outputs
    // remain live inside the parent RuntimeContext at this point.
    EXPECT_EQ(alloc.allocate_calls(), 8u);
    EXPECT_EQ(alloc.free_calls(), 6u);
    EXPECT_EQ(alloc.allocated_count(), 2u);
  }
  EXPECT_EQ(alloc.allocate_calls(), 8u);
  EXPECT_EQ(alloc.free_calls(), 8u);
  EXPECT_EQ(alloc.allocated_count(), 0u);
}

TEST(RunLoopWithSequenceState, SequenceOnlyStateGrowsPerIteration) {
  RuntimeContext rt(KernelContext(DefaultOpset(13)));
  rt.Set("M", Tensor::FromInt64("M", {}, {3}));
  rt.Set("cond", Tensor::FromBool("cond", {}, {1}));
  rt.PutSequence("seq_init", Sequence("seq_init", static_cast<int32_t>(DataType::FLOAT), {}));

  RunNode(MakeLoopNode({"M", "cond", "seq_init"}, {"seq_out"}, BuildSequenceLoopBody()), rt);

  ASSERT_TRUE(rt.HasSequence("seq_out"));
  const Sequence &seq = rt.GetSequence("seq_out");
  ASSERT_EQ(seq.size(), 3u);
  EXPECT_EQ(seq.elem_type, static_cast<int32_t>(DataType::FLOAT));
  for (std::size_t i = 0; i < seq.size(); ++i) {
    ASSERT_EQ(seq.at(i).shape, std::vector<int64_t>({1}));
    EXPECT_FLOAT_EQ(seq.at(i).AsFloat()[0], static_cast<float>(i));
  }
}

TEST(RunLoopWithSequenceState, SequenceOnlyStateAllocatorBacksIterAndCondScalars) {
  CountingAllocator alloc(/*capacity=*/2);
  {
    TensorMap tensors;
    tensors.emplace("M", Tensor::FromInt64("M", {}, {3}));
    tensors.emplace("cond", Tensor::FromBool("cond", {}, {1}));
    RuntimeContext rt(KernelContext(DefaultOpset(13)), std::move(tensors),
                      core::runtime::RuntimeContextOptions{.allocator = &alloc});
    rt.PutSequence("seq_init", Sequence("seq_init", static_cast<int32_t>(DataType::FLOAT), {}));

    RunNode(MakeLoopNode({"M", "cond", "seq_init"}, {"seq_out"}, BuildSequenceLoopBody()), rt);

    // 3 iterations x 2 transient scalar inputs (iter, cond_in) = 6
    // allocations total, and the sequence-only loop carries no allocator-backed
    // tensor outputs, so every slot has already been freed.
    EXPECT_EQ(alloc.allocate_calls(), 6u);
    EXPECT_EQ(alloc.free_calls(), 6u);
    EXPECT_EQ(alloc.allocated_count(), 0u);
  }
  EXPECT_EQ(alloc.allocate_calls(), 6u);
  EXPECT_EQ(alloc.free_calls(), 6u);
  EXPECT_EQ(alloc.allocated_count(), 0u);
}

// RunLoopWithSequenceState: the loop ``cond`` input set to ``false``
// short-circuits the loop, so the sequence loop-carried output equals
// the (non-empty) input sequence unchanged.
TEST(RunLoopWithSequenceState, ZeroTripReturnsInputSequence) {
  RuntimeContext rt(KernelContext(DefaultOpset(13)));
  rt.Set("M", Tensor::FromInt64("M", {}, {5}));
  rt.Set("cond", Tensor::FromBool("cond", {}, {0}));
  rt.PutSequence("seq_init", Sequence("seq_init", static_cast<int32_t>(DataType::FLOAT),
                                      {Tensor::FromFloat("e0", {1}, {9.0f})}));

  RunNode(MakeLoopNode({"M", "cond", "seq_init"}, {"seq_out"}, BuildSequenceLoopBody()), rt);

  ASSERT_TRUE(rt.HasSequence("seq_out"));
  const Sequence &seq = rt.GetSequence("seq_out");
  ASSERT_EQ(seq.size(), 1u);
  EXPECT_FLOAT_EQ(seq.at(0).AsFloat()[0], 9.0f);
}

// RunLoopWithSequenceState: mixed state where a tensor accumulator and a
// sequence are both loop-carried, plus a per-iteration scan output. The
// presence of the sequence state routes the Loop through
// RunLoopWithSequenceState (rather than kernel::Loop), and this exercises
// the tensor-state, sequence-state and scan-stacking paths together.
TEST(RunLoopWithSequenceState, MixedTensorSequenceAndScanOutputs) {
  RuntimeContext rt(KernelContext(DefaultOpset(13)));
  rt.Set("M", Tensor::FromInt64("M", {}, {3}));
  rt.Set("cond", Tensor::FromBool("cond", {}, {1}));
  rt.Set("acc_init", Tensor::FromFloat("acc_init", {}, {0.0f}));
  rt.PutSequence("seq_init", Sequence("seq_init", static_cast<int32_t>(DataType::FLOAT), {}));

  RunNode(MakeLoopNode({"M", "cond", "acc_init", "seq_init"}, {"acc_final", "seq_final", "scan"},
                       BuildMixedSequenceLoopBody()),
          rt);

  ASSERT_TRUE(rt.Has("acc_final"));
  EXPECT_FLOAT_EQ(rt.Get("acc_final").AsFloat()[0], 3.0f);

  ASSERT_TRUE(rt.HasSequence("seq_final"));
  const Sequence &seq = rt.GetSequence("seq_final");
  ASSERT_EQ(seq.size(), 3u);
  for (std::size_t i = 0; i < seq.size(); ++i) {
    EXPECT_FLOAT_EQ(seq.at(i).AsFloat()[0], static_cast<float>(i));
  }

  ASSERT_TRUE(rt.Has("scan"));
  const Tensor &scan = rt.Get("scan");
  ASSERT_EQ(scan.shape, (std::vector<int64_t>{3}));
  const float *scan_data = scan.AsFloat();
  EXPECT_FLOAT_EQ(scan_data[0], 1.0f);
  EXPECT_FLOAT_EQ(scan_data[1], 2.0f);
  EXPECT_FLOAT_EQ(scan_data[2], 3.0f);
}

// Same as MixedTensorSequenceAndScanOutputs but with a SimpleRawBufferAllocator
// attached to the RuntimeContext. Verifies that subgraph contexts do not inherit
// the parent allocator (preventing double-free of loop-carried outputs), while
// the transient iter/cond scalar bindings and the final loop outputs are backed
// by the parent allocator.
TEST(RunLoopWithSequenceState, MixedTensorSequenceAndScanOutputsWithAllocator) {
  // Two simultaneous slots are needed: the transient iter/cond allocations are
  // released before the final outputs are stored. Input tensors are set before
  // attaching the allocator so they stay inline and do not consume allocator
  // slots.
  constexpr size_t kAllocatorSlotCapacity = 2;
  core::runtime::SimpleRawBufferAllocator alloc(kAllocatorSlotCapacity);

  TensorMap tensors;
  tensors.emplace("M", Tensor::FromInt64("M", {}, {3}));
  tensors.emplace("cond", Tensor::FromBool("cond", {}, {1}));
  tensors.emplace("acc_init", Tensor::FromFloat("acc_init", {}, {0.0f}));
  RuntimeContext rt(KernelContext(DefaultOpset(13)), std::move(tensors),
                    core::runtime::RuntimeContextOptions{.allocator = &alloc});
  rt.PutSequence("seq_init", Sequence("seq_init", static_cast<int32_t>(DataType::FLOAT), {}));

  RunNode(MakeLoopNode({"M", "cond", "acc_init", "seq_init"}, {"acc_final", "seq_final", "scan"},
                       BuildMixedSequenceLoopBody()),
          rt);

  ASSERT_TRUE(rt.Has("acc_final"));
  EXPECT_FLOAT_EQ(rt.Get("acc_final").AsFloat()[0], 3.0f);

  ASSERT_TRUE(rt.HasSequence("seq_final"));
  const Sequence &seq = rt.GetSequence("seq_final");
  ASSERT_EQ(seq.size(), 3u);
  for (std::size_t i = 0; i < seq.size(); ++i) {
    EXPECT_FLOAT_EQ(seq.at(i).AsFloat()[0], static_cast<float>(i));
  }

  ASSERT_TRUE(rt.Has("scan"));
  const Tensor &scan = rt.Get("scan");
  ASSERT_EQ(scan.shape, (std::vector<int64_t>{3}));
  const float *scan_data = scan.AsFloat();
  EXPECT_FLOAT_EQ(scan_data[0], 1.0f);
  EXPECT_FLOAT_EQ(scan_data[1], 2.0f);
  EXPECT_FLOAT_EQ(scan_data[2], 3.0f);
}

TEST(RunNodes, SliceTensorAlongAxisKeepsInputAllocator) {
  core::runtime::SimpleRawBufferAllocator alloc(2);
  const Tensor input = Tensor::FromFloat("x", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, &alloc);

  ASSERT_TRUE(input.has_allocation());
  EXPECT_EQ(alloc.allocated_count(), 1u);

  const Tensor slice = SliceTensorAlongAxis(input, 0, 1, "slice_row");

  EXPECT_TRUE(slice.has_allocation());
  EXPECT_EQ(slice.allocation_owner(), input.allocation_owner());
  EXPECT_EQ(alloc.allocated_count(), 2u);
  ASSERT_EQ(slice.shape, (std::vector<int64_t>{3}));
  const float *values = slice.AsFloat();
  EXPECT_FLOAT_EQ(values[0], 4.0f);
  EXPECT_FLOAT_EQ(values[1], 5.0f);
  EXPECT_FLOAT_EQ(values[2], 6.0f);
}

TEST(RunNodes, SliceTensorAlongAxisUsesInlineStorageWithoutAllocator) {
  const Tensor input = Tensor::FromFloat("x", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

  ASSERT_FALSE(input.has_allocation());

  const Tensor slice = SliceTensorAlongAxis(input, 0, 0, "slice_row");

  EXPECT_FALSE(slice.has_allocation());
  EXPECT_EQ(slice.name, "x_slice");
  ASSERT_EQ(slice.shape, (std::vector<int64_t>{3}));
  EXPECT_EQ(slice.data.size(), 3u * sizeof(float));
  const float *values = slice.AsFloat();
  EXPECT_FLOAT_EQ(values[0], 1.0f);
  EXPECT_FLOAT_EQ(values[1], 2.0f);
  EXPECT_FLOAT_EQ(values[2], 3.0f);
}

TEST(RunModel, ScanNodeRunsBodySubgraph) {
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(18);

  GraphProto *g = model.add_graph();
  g->set_name("main");
  NodeProto *scan = g->add_node();
  scan->set_op_type("Scan");
  scan->add_input("state0");
  scan->add_input("x");
  scan->add_output("state_final");
  scan->add_output("y");
  AttributeProto *num_attr = scan->add_attribute();
  num_attr->set_name("num_scan_inputs");
  num_attr->set_type(AttributeProto::AttributeType::INT);
  num_attr->set_i(1);

  AttributeProto *body_attr = scan->add_attribute();
  body_attr->set_name("body");
  body_attr->set_type(AttributeProto::AttributeType::GRAPH);
  GraphProto *body = body_attr->add_g();
  body->set_name("scan_body");
  body->add_input()->set_name("state_in");
  body->add_input()->set_name("x_in");
  NodeProto *add = body->add_node();
  add->set_op_type("Add");
  add->add_input("state_in");
  add->add_input("x_in");
  add->add_output("state_out");
  body->add_output()->set_name("state_out");
  body->add_output()->set_name("state_out");

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("state0", Tensor::FromFloat("state0", {}, {0.0f}));
  rt.Set("x", Tensor::FromFloat("x", {3}, {1.0f, 2.0f, 3.0f}));

  RunModelViaSession(model, rt);

  ASSERT_TRUE(rt.Has("state_final"));
  ASSERT_TRUE(rt.Has("y"));
  EXPECT_FLOAT_EQ(rt.Get("state_final").AsFloat()[0], 6.0f);
  ASSERT_EQ(rt.Get("y").shape, (std::vector<int64_t>{3}));
  const float *y = rt.Get("y").AsFloat();
  EXPECT_FLOAT_EQ(y[0], 1.0f);
  EXPECT_FLOAT_EQ(y[1], 3.0f);
  EXPECT_FLOAT_EQ(y[2], 6.0f);
}

// Mirrors backend test ``test_scan_sum``: opset 8 batched form of Scan
// where every state and scan input/output carries an outer batch dim of
// size 1 and the (always-empty here) ``sequence_lens`` placeholder
// occupies node.input(0).
//
//   sum_in = Add(sum_in, next), scan_out = Identity(sum_out)
//   initial=[[0,0]] [1,2], x=[[[1,2],[3,4],[5,6]]] [1,3,2]
//   y=[[9,12]] [1,2], z=[[[1,2],[4,6],[9,12]]] [1,3,2]
TEST(RunModel, ScanOpset8NodeRunsBodySubgraphWithBatchDim) {
  ModelProto model;
  model.set_ir_version(3);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(8);

  GraphProto *g = model.add_graph();
  g->set_name("main");
  NodeProto *scan = g->add_node();
  scan->set_op_type("Scan");
  // Empty placeholder for sequence_lens, then initial state, then scan input.
  scan->add_input("");
  scan->add_input("initial");
  scan->add_input("x");
  scan->add_output("y");
  scan->add_output("z");
  AttributeProto *num_attr = scan->add_attribute();
  num_attr->set_name("num_scan_inputs");
  num_attr->set_type(AttributeProto::AttributeType::INT);
  num_attr->set_i(1);

  AttributeProto *body_attr = scan->add_attribute();
  body_attr->set_name("body");
  body_attr->set_type(AttributeProto::AttributeType::GRAPH);
  GraphProto *body = body_attr->add_g();
  body->set_name("scan_body");
  body->add_input()->set_name("sum_in");
  body->add_input()->set_name("next");
  NodeProto *add = body->add_node();
  add->set_op_type("Add");
  add->add_input("sum_in");
  add->add_input("next");
  add->add_output("sum_out");
  NodeProto *id = body->add_node();
  id->set_op_type("Identity");
  id->add_input("sum_out");
  id->add_output("scan_out");
  body->add_output()->set_name("sum_out");
  body->add_output()->set_name("scan_out");

  RuntimeContext rt(KernelContext(DefaultOpset(8)));
  rt.Set("initial", Tensor::FromFloat("initial", {1, 2}, {0.0f, 0.0f}));
  rt.Set("x", Tensor::FromFloat("x", {1, 3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));

  RunModelViaSession(model, rt);

  ASSERT_TRUE(rt.Has("y"));
  ASSERT_TRUE(rt.Has("z"));

  const Tensor &y = rt.Get("y");
  ASSERT_EQ(y.shape, (std::vector<int64_t>{1, 2}));
  const float *y_ptr = y.AsFloat();
  EXPECT_FLOAT_EQ(y_ptr[0], 9.0f);
  EXPECT_FLOAT_EQ(y_ptr[1], 12.0f);

  const Tensor &z = rt.Get("z");
  ASSERT_EQ(z.shape, (std::vector<int64_t>{1, 3, 2}));
  const float *z_ptr = z.AsFloat();
  EXPECT_FLOAT_EQ(z_ptr[0], 1.0f);
  EXPECT_FLOAT_EQ(z_ptr[1], 2.0f);
  EXPECT_FLOAT_EQ(z_ptr[2], 4.0f);
  EXPECT_FLOAT_EQ(z_ptr[3], 6.0f);
  EXPECT_FLOAT_EQ(z_ptr[4], 9.0f);
  EXPECT_FLOAT_EQ(z_ptr[5], 12.0f);
}

TEST(RunNodes, RunNodeLinearAttentionFromDispatchTable) {
  // Minimal test: B=1, T=1, q_num_heads=1, kv_num_heads=1, d_k=d_v=2.
  // query/key/value each have shape (1, 1, 2).
  RuntimeContext rt(KernelContext(DefaultOpset(27)));
  rt.tensors()["query"] = Tensor::FromFloat("query", {1, 1, 2}, {1.0f, 0.0f});
  rt.tensors()["key"] = Tensor::FromFloat("key", {1, 1, 2}, {1.0f, 0.0f});
  rt.tensors()["value"] = Tensor::FromFloat("value", {1, 1, 2}, {0.5f, 0.5f});

  NodeProto node =
      MakeNode("LinearAttention", {"query", "key", "value"}, {"output", "present_state"});
  AttributeProto *rule_attr = node.add_attribute();
  rule_attr->set_name("update_rule");
  rule_attr->set_type(AttributeProto::AttributeType::STRING);
  rule_attr->set_s("linear");
  AttributeProto *qh_attr = node.add_attribute();
  qh_attr->set_name("q_num_heads");
  qh_attr->set_type(AttributeProto::AttributeType::INT);
  qh_attr->set_i(1);
  AttributeProto *kvh_attr = node.add_attribute();
  kvh_attr->set_name("kv_num_heads");
  kvh_attr->set_type(AttributeProto::AttributeType::INT);
  kvh_attr->set_i(1);
  RunNode(node, rt);

  const Tensor &output = rt.tensors().at("output");
  EXPECT_EQ(output.shape, (std::vector<int64_t>{1, 1, 2}));
  const Tensor &present = rt.tensors().at("present_state");
  EXPECT_EQ(present.shape, (std::vector<int64_t>{1, 1, 2, 2}));
}

TEST(RunNodes, RunNodeFlexAttentionFromDispatchTable) {
  // Minimal test: B=1, q_num_heads=kv_num_heads=1, q_seq=kv_seq=2, head_size=2.
  // Q/K/V each have shape (1, 1, 2, 2).
  RuntimeContext rt(KernelContext(DefaultOpset(13)));
  rt.tensors()["Q"] = Tensor::FromFloat("Q", {1, 1, 2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});
  rt.tensors()["K"] = Tensor::FromFloat("K", {1, 1, 2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});
  rt.tensors()["V"] = Tensor::FromFloat("V", {1, 1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});

  NodeProto node = MakeNode("FlexAttention", {"Q", "K", "V"}, {"Y"}, "ai.onnx.preview");
  RunNode(node, rt);

  const Tensor &Y = rt.tensors().at("Y");
  EXPECT_EQ(Y.shape, (std::vector<int64_t>{1, 1, 2, 2}));
  EXPECT_EQ(Y.data_type, static_cast<int32_t>(core::runtime::DataType::FLOAT));
}

TEST(RunNodes, RunNodeGRUFromDispatchTable) {
  // Single-step (seq_length=1) GRU with X/W/R only: requests Y_h as the
  // only output via an empty Y output name, mirroring the ``gru_defaults``
  // backend test case (batch=3, input=2, hidden=5).
  RuntimeContext rt(KernelContext(DefaultOpset(14)));

  constexpr int64_t kSeqLength = 1;
  constexpr int64_t kBatch = 3;
  constexpr int64_t kInput = 2;
  constexpr int64_t kHidden = 5;
  constexpr int64_t kNumGates = 3;
  constexpr float kWeightScale = 0.1f;

  rt.tensors()["X"] = Tensor::FromFloat("X", {kSeqLength, kBatch, kInput}, {1, 2, 3, 4, 5, 6});
  std::vector<float> w_data(static_cast<size_t>(kNumGates * kHidden * kInput), kWeightScale);
  std::vector<float> r_data(static_cast<size_t>(kNumGates * kHidden * kHidden), kWeightScale);
  rt.tensors()["W"] = Tensor::FromFloat("W", {1, kNumGates * kHidden, kInput}, w_data);
  rt.tensors()["R"] = Tensor::FromFloat("R", {1, kNumGates * kHidden, kHidden}, r_data);

  NodeProto node = MakeNode("GRU", {"X", "W", "R"}, {"", "Y_h"});
  AttributeProto *hs = node.add_attribute();
  hs->set_name("hidden_size");
  hs->set_type(AttributeProto::AttributeType::INT);
  hs->set_i(kHidden);

  RunNode(node, rt);

  // Y is suppressed (empty output name) so it must not appear in the tensors map.
  EXPECT_EQ(rt.tensors().find("Y"), rt.tensors().end());

  const Tensor &y_h = rt.tensors().at("Y_h");
  EXPECT_EQ(y_h.shape, (std::vector<int64_t>{1, kBatch, kHidden}));
  EXPECT_EQ(y_h.data_type, static_cast<int32_t>(core::runtime::DataType::FLOAT));

  // Compare against the kernel's direct output to validate dispatch-time
  // wiring of inputs, attributes and outputs.
  const onnx_kernels::kernel::GRU gru_kernel(rt.kernel_ctx());
  auto [y_ref, y_h_ref] =
      gru_kernel(rt.tensors().at("X"), rt.tensors().at("W"), rt.tensors().at("R"));
  (void)y_ref;
  ASSERT_EQ(y_h.element_count(), y_h_ref.element_count());
  for (int64_t i = 0; i < y_h.element_count(); ++i) {
    EXPECT_FLOAT_EQ(y_h.AsFloat()[i], y_h_ref.AsFloat()[i]);
  }
}

TEST(RunNodes, RunNodeLSTMFromDispatchTableUniformSequenceLens) {
  // ``sequence_lens`` is accepted when every entry equals ``seq_length``
  // (no-op masking); the dispatch must produce the same outputs as the
  // 3-input form above. This mirrors the ``test_cc_lstm_with_peepholes``
  // backend case (which passes a uniform ``sequence_lens``).
  RuntimeContext rt(KernelContext(DefaultOpset(14)));

  constexpr int64_t kSeqLength = 1;
  constexpr int64_t kBatch = 2;
  constexpr int64_t kInput = 4;
  constexpr int64_t kHidden = 3;
  constexpr int64_t kNumGates = 4;
  constexpr int64_t kNumPeepholes = 3;
  constexpr float kWeightScale = 0.1f;

  rt.tensors()["X"] =
      Tensor::FromFloat("X", {kSeqLength, kBatch, kInput}, {1, 2, 3, 4, 5, 6, 7, 8});
  std::vector<float> w_data(static_cast<size_t>(kNumGates * kHidden * kInput), kWeightScale);
  std::vector<float> r_data(static_cast<size_t>(kNumGates * kHidden * kHidden), kWeightScale);
  std::vector<float> b_data(static_cast<size_t>(2 * kNumGates * kHidden), 0.0f);
  std::vector<float> h0_data(static_cast<size_t>(kBatch * kHidden), 0.0f);
  std::vector<float> c0_data(static_cast<size_t>(kBatch * kHidden), 0.0f);
  std::vector<float> p_data(static_cast<size_t>(kNumPeepholes * kHidden), kWeightScale);
  rt.tensors()["W"] = Tensor::FromFloat("W", {1, kNumGates * kHidden, kInput}, w_data);
  rt.tensors()["R"] = Tensor::FromFloat("R", {1, kNumGates * kHidden, kHidden}, r_data);
  rt.tensors()["B"] = Tensor::FromFloat("B", {1, 2 * kNumGates * kHidden}, b_data);
  rt.tensors()["sequence_lens"] =
      Tensor::FromInt32("sequence_lens", {kBatch},
                        {static_cast<int32_t>(kSeqLength), static_cast<int32_t>(kSeqLength)});
  rt.tensors()["initial_h"] = Tensor::FromFloat("initial_h", {1, kBatch, kHidden}, h0_data);
  rt.tensors()["initial_c"] = Tensor::FromFloat("initial_c", {1, kBatch, kHidden}, c0_data);
  rt.tensors()["P"] = Tensor::FromFloat("P", {1, kNumPeepholes * kHidden}, p_data);

  NodeProto node = MakeNode(
      "LSTM", {"X", "W", "R", "B", "sequence_lens", "initial_h", "initial_c", "P"}, {"", "Y_h"});
  AttributeProto *hs = node.add_attribute();
  hs->set_name("hidden_size");
  hs->set_type(AttributeProto::AttributeType::INT);
  hs->set_i(kHidden);

  RunNode(node, rt);

  const Tensor &y_h = rt.tensors().at("Y_h");
  EXPECT_EQ(y_h.shape, (std::vector<int64_t>{1, kBatch, kHidden}));
  EXPECT_EQ(y_h.data_type, static_cast<int32_t>(core::runtime::DataType::FLOAT));

  // Non-uniform ``sequence_lens`` is still rejected.
  rt.tensors()["sequence_lens"] =
      Tensor::FromInt32("sequence_lens", {kBatch}, {static_cast<int32_t>(kSeqLength), 0});
  EXPECT_THROW(RunNode(node, rt), std::invalid_argument);
}

TEST(RunNodes, RunNodeLSTMFromDispatchTable) {
  // Single-step (seq_length=1) LSTM with X/W/R only: requests Y_h as the
  // only output via an empty Y output name, mirroring the ``lstm_defaults``
  // backend test case (batch=3, input=2, hidden=3).
  RuntimeContext rt(KernelContext(DefaultOpset(14)));

  constexpr int64_t kSeqLength = 1;
  constexpr int64_t kBatch = 3;
  constexpr int64_t kInput = 2;
  constexpr int64_t kHidden = 3;
  constexpr int64_t kNumGates = 4;
  constexpr float kWeightScale = 0.1f;

  rt.tensors()["X"] = Tensor::FromFloat("X", {kSeqLength, kBatch, kInput}, {1, 2, 3, 4, 5, 6});
  std::vector<float> w_data(static_cast<size_t>(kNumGates * kHidden * kInput), kWeightScale);
  std::vector<float> r_data(static_cast<size_t>(kNumGates * kHidden * kHidden), kWeightScale);
  rt.tensors()["W"] = Tensor::FromFloat("W", {1, kNumGates * kHidden, kInput}, w_data);
  rt.tensors()["R"] = Tensor::FromFloat("R", {1, kNumGates * kHidden, kHidden}, r_data);

  NodeProto node = MakeNode("LSTM", {"X", "W", "R"}, {"", "Y_h"});
  AttributeProto *hs = node.add_attribute();
  hs->set_name("hidden_size");
  hs->set_type(AttributeProto::AttributeType::INT);
  hs->set_i(kHidden);

  RunNode(node, rt);

  // Y is suppressed (empty output name) so it must not appear in the tensors map.
  EXPECT_EQ(rt.tensors().find("Y"), rt.tensors().end());

  const Tensor &y_h = rt.tensors().at("Y_h");
  EXPECT_EQ(y_h.shape, (std::vector<int64_t>{1, kBatch, kHidden}));
  EXPECT_EQ(y_h.data_type, static_cast<int32_t>(core::runtime::DataType::FLOAT));

  // Compare against the kernel's direct output to validate dispatch-time
  // wiring of inputs, attributes and outputs.
  const onnx_kernels::kernel::LSTM lstm_kernel(rt.kernel_ctx());
  auto [y_ref, y_h_ref, y_c_ref] =
      lstm_kernel(rt.tensors().at("X"), rt.tensors().at("W"), rt.tensors().at("R"));
  (void)y_ref;
  (void)y_c_ref;
  ASSERT_EQ(y_h.element_count(), y_h_ref.element_count());
  for (int64_t i = 0; i < y_h.element_count(); ++i) {
    EXPECT_FLOAT_EQ(y_h.AsFloat()[i], y_h_ref.AsFloat()[i]);
  }
}

TEST(RunNodes, RunNodeSequenceConstructAndQueriesFromDispatchTable) {
  // Build a sequence of three tensors with SequenceConstruct, then
  // dispatch SequenceLength, SequenceAt and ConcatFromSequence and
  // check that the sequence-typed edge flows through the runtime
  // context's sequence map.
  RuntimeContext rt(KernelContext(DefaultOpset(11)));
  rt.tensors()["a"] = Tensor::FromFloat("a", {2}, {1.0f, 2.0f});
  rt.tensors()["b"] = Tensor::FromFloat("b", {2}, {3.0f, 4.0f});
  rt.tensors()["c"] = Tensor::FromFloat("c", {2}, {5.0f, 6.0f});
  rt.tensors()["pos1"] = Tensor::FromInt64("pos1", {}, {1});

  RunNode(MakeNode("SequenceConstruct", {"a", "b", "c"}, {"seq"}), rt);
  ASSERT_TRUE(rt.HasSequence("seq"));
  EXPECT_EQ(rt.GetSequence("seq").size(), 3u);
  EXPECT_EQ(rt.GetSequence("seq").elem_type, static_cast<int32_t>(core::runtime::DataType::FLOAT));

  RunNode(MakeNode("SequenceLength", {"seq"}, {"n"}), rt);
  const Tensor &n = rt.tensors().at("n");
  EXPECT_EQ(n.data_type, static_cast<int32_t>(core::runtime::DataType::INT64));
  ASSERT_EQ(n.element_count(), 1);
  EXPECT_EQ(n.AsInt64()[0], 3);

  RunNode(MakeNode("SequenceAt", {"seq", "pos1"}, {"middle"}), rt);
  const Tensor &middle = rt.tensors().at("middle");
  EXPECT_EQ(middle.shape, std::vector<int64_t>({2}));
  EXPECT_FLOAT_EQ(middle.AsFloat()[0], 3.0f);
  EXPECT_FLOAT_EQ(middle.AsFloat()[1], 4.0f);

  NodeProto concat = MakeNode("ConcatFromSequence", {"seq"}, {"flat"});
  AttributeProto *axis = concat.add_attribute();
  axis->set_name("axis");
  axis->set_type(AttributeProto::INT);
  axis->set_i(0);
  RunNode(concat, rt);
  const Tensor &flat = rt.tensors().at("flat");
  EXPECT_EQ(flat.shape, std::vector<int64_t>({6}));
  const float *p = flat.AsFloat();
  for (int i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(p[i], static_cast<float>(i + 1));
  }
}

TEST(RunNodes, RunNodeSequenceEmptyInsertEraseFromDispatchTable) {
  RuntimeContext rt(KernelContext(DefaultOpset(11)));
  rt.tensors()["a"] = Tensor::FromFloat("a", {1}, {7.0f});
  rt.tensors()["b"] = Tensor::FromFloat("b", {1}, {8.0f});

  NodeProto empty = MakeNode("SequenceEmpty", {}, {"seq0"});
  AttributeProto *dtype = empty.add_attribute();
  dtype->set_name("dtype");
  dtype->set_type(AttributeProto::INT);
  dtype->set_i(static_cast<int64_t>(core::runtime::DataType::FLOAT));
  RunNode(empty, rt);
  ASSERT_TRUE(rt.HasSequence("seq0"));
  EXPECT_TRUE(rt.GetSequence("seq0").empty());

  RunNode(MakeNode("SequenceInsert", {"seq0", "a"}, {"seq1"}), rt);
  EXPECT_EQ(rt.GetSequence("seq1").size(), 1u);
  RunNode(MakeNode("SequenceInsert", {"seq1", "b"}, {"seq2"}), rt);
  EXPECT_EQ(rt.GetSequence("seq2").size(), 2u);

  RunNode(MakeNode("SequenceErase", {"seq2"}, {"seq3"}), rt);
  EXPECT_EQ(rt.GetSequence("seq3").size(), 1u);
  EXPECT_FLOAT_EQ(rt.GetSequence("seq3").at(0).AsFloat()[0], 7.0f);
}

TEST(RunNodes, RunNodeSplitToSequenceFromDispatchTable) {
  RuntimeContext rt(KernelContext(DefaultOpset(11)));
  rt.tensors()["x"] = Tensor::FromFloat("x", {3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

  NodeProto split = MakeNode("SplitToSequence", {"x"}, {"out"});
  AttributeProto *keepdims = split.add_attribute();
  keepdims->set_name("keepdims");
  keepdims->set_type(AttributeProto::INT);
  keepdims->set_i(0);
  RunNode(split, rt);

  ASSERT_TRUE(rt.HasSequence("out"));
  const auto &seq = rt.GetSequence("out");
  EXPECT_EQ(seq.size(), 3u);
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(seq.at(i).shape, std::vector<int64_t>({2}));
    EXPECT_FLOAT_EQ(seq.at(i).AsFloat()[0], static_cast<float>(2 * i + 1));
    EXPECT_FLOAT_EQ(seq.at(i).AsFloat()[1], static_cast<float>(2 * i + 2));
  }
}

TEST(RuntimeContextCollectExternalInputs, FlatNodes) {
  std::vector<NodeProto> nodes;
  nodes.push_back(MakeNode("Mul", {"x", "y"}, {"t"}));
  nodes.push_back(MakeNode("Sub", {"t", "z"}, {"out"}));
  nodes.push_back(MakeNode("Add", {"out", "x"}, {"final"}));

  auto inputs = core::runtime::RuntimeSession::CollectExternalInputs(nodes);
  EXPECT_EQ(inputs, std::vector<std::string>({"x", "y", "z"}));
}

TEST(RuntimeContextCollectExternalInputs, EmptyNodes) {
  std::vector<NodeProto> nodes;
  EXPECT_TRUE(core::runtime::RuntimeSession::CollectExternalInputs(nodes).empty());
}

TEST(RuntimeContextCollectExternalInputs, SkipsEmptyAndProducedNames) {
  std::vector<NodeProto> nodes;
  // Optional input "" must be ignored; output "t" produced internally
  // must not be reported as external.
  nodes.push_back(MakeNode("Resize", {"X", "", "scales"}, {"t"}));
  nodes.push_back(MakeNode("Abs", {"t"}, {"Y"}));

  auto inputs = core::runtime::RuntimeSession::CollectExternalInputs(nodes);
  EXPECT_EQ(inputs, std::vector<std::string>({"X", "scales"}));
}

TEST(RuntimeContextCollectExternalInputs, SubgraphCapturesOuterValues) {
  // Build an If node whose then_branch reads "cap1" from the outer
  // scope and whose else_branch reads "cap2"; "cond" feeds the If
  // input and "produced" is produced by an earlier node in the set.
  GraphProto then_branch;
  *then_branch.add_node() = MakeNode("Add", {"cap1", "produced"}, {"then_out"});
  then_branch.add_output()->set_name("then_out");

  GraphProto else_branch;
  *else_branch.add_node() = MakeNode("Add", {"cap2", "produced"}, {"else_out"});
  else_branch.add_output()->set_name("else_out");

  NodeProto if_node = MakeNode("If", {"cond"}, {"y"});
  AttributeProto *attr_then = if_node.add_attribute();
  attr_then->set_name("then_branch");
  attr_then->set_type(AttributeProto::AttributeType::GRAPH);
  *attr_then->mutable_g() = then_branch;
  AttributeProto *attr_else = if_node.add_attribute();
  attr_else->set_name("else_branch");
  attr_else->set_type(AttributeProto::AttributeType::GRAPH);
  *attr_else->mutable_g() = else_branch;

  std::vector<NodeProto> nodes;
  nodes.push_back(MakeNode("Identity", {"x"}, {"produced"}));
  nodes.push_back(if_node);

  auto inputs = core::runtime::RuntimeSession::CollectExternalInputs(nodes);
  // "produced" is produced by the outer set and must not be reported.
  // Order is first-seen.
  EXPECT_EQ(inputs, std::vector<std::string>({"x", "cond", "cap1", "cap2"}));
}

TEST(RuntimeContextCollectExternalInputs, SubgraphLocalNamesShadowOuter) {
  // Subgraph defines its own formal input "x", an initializer "k",
  // and produces "tmp" internally — none of these should be reported.
  // It additionally reads "outer_only" from the outer scope.
  GraphProto body;
  body.add_input()->set_name("x");
  TensorProto *init = body.add_initializer();
  init->set_name("k");
  init->set_data_type(static_cast<int32_t>(DataType::FLOAT));
  *body.add_node() = MakeNode("Add", {"x", "k"}, {"tmp"});
  *body.add_node() = MakeNode("Mul", {"tmp", "outer_only"}, {"body_out"});
  body.add_output()->set_name("body_out");

  NodeProto loop = MakeNode("Loop", {"M", "cond"}, {"y"});
  AttributeProto *attr = loop.add_attribute();
  attr->set_name("body");
  attr->set_type(AttributeProto::AttributeType::GRAPH);
  *attr->mutable_g() = body;

  std::vector<NodeProto> nodes = {loop};
  auto inputs = core::runtime::RuntimeSession::CollectExternalInputs(nodes);
  EXPECT_EQ(inputs, std::vector<std::string>({"M", "cond", "outer_only"}));
}

TEST(RuntimeContextCollectExternalInputs, NestedSubgraphCaptures) {
  // Outer If holds an inner If whose then_branch reads "deep".
  GraphProto inner_then;
  *inner_then.add_node() = MakeNode("Identity", {"deep"}, {"inner_out"});
  inner_then.add_output()->set_name("inner_out");
  GraphProto inner_else;
  *inner_else.add_node() = MakeNode("Identity", {"deep"}, {"inner_out"});
  inner_else.add_output()->set_name("inner_out");

  NodeProto inner_if = MakeNode("If", {"inner_cond"}, {"middle"});
  AttributeProto *a1 = inner_if.add_attribute();
  a1->set_name("then_branch");
  a1->set_type(AttributeProto::AttributeType::GRAPH);
  *a1->mutable_g() = inner_then;
  AttributeProto *a2 = inner_if.add_attribute();
  a2->set_name("else_branch");
  a2->set_type(AttributeProto::AttributeType::GRAPH);
  *a2->mutable_g() = inner_else;

  GraphProto outer_then;
  *outer_then.add_node() = inner_if;
  *outer_then.add_node() = MakeNode("Identity", {"middle"}, {"outer_out"});
  outer_then.add_output()->set_name("outer_out");
  GraphProto outer_else;
  *outer_else.add_node() = MakeNode("Identity", {"middle"}, {"outer_out"});
  outer_else.add_output()->set_name("outer_out");

  NodeProto outer_if = MakeNode("If", {"outer_cond"}, {"y"});
  AttributeProto *b1 = outer_if.add_attribute();
  b1->set_name("then_branch");
  b1->set_type(AttributeProto::AttributeType::GRAPH);
  *b1->mutable_g() = outer_then;
  AttributeProto *b2 = outer_if.add_attribute();
  b2->set_name("else_branch");
  b2->set_type(AttributeProto::AttributeType::GRAPH);
  *b2->mutable_g() = outer_else;

  std::vector<NodeProto> nodes = {outer_if};
  auto inputs = core::runtime::RuntimeSession::CollectExternalInputs(nodes);
  // outer_cond is read by the outer If node itself.
  // Inside outer_then: inner_if introduces inner_cond, and its branches
  // capture "deep" from above.
  // Inside outer_else: the lone Identity reads "middle", which is not
  // produced anywhere in outer_else (only in outer_then via inner_if),
  // so it is captured from the outer scope.
  EXPECT_EQ(inputs, std::vector<std::string>({"outer_cond", "inner_cond", "deep", "middle"}));
}

TEST(RuntimeContextCollectExternalInputs, DeduplicatesOrdering) {
  std::vector<NodeProto> nodes;
  nodes.push_back(MakeNode("Add", {"a", "b"}, {"u"}));
  nodes.push_back(MakeNode("Mul", {"a", "u"}, {"v"})); // re-references "a"
  nodes.push_back(MakeNode("Sub", {"b", "v"}, {"w"})); // re-references "b"

  auto inputs = core::runtime::RuntimeSession::CollectExternalInputs(nodes);
  EXPECT_EQ(inputs, std::vector<std::string>({"a", "b"}));
}

// ---------------------------------------------------------------------------
// RuntimeContext isolation invariants when running a local function or a
// subgraph. See issue #2157.
// ---------------------------------------------------------------------------

// A model-local function must be invoked with an isolated tensor map: only
// its formal inputs (bound to the caller's actuals) are visible inside the
// function body. Names that exist in the caller's tensor map but are not
// passed as a function input must NOT be visible from inside the function.
// The construction-time ``kernel_ctx()`` is still shared so the function's
// nodes are dispatched against the same opset as the caller.
TEST(RunModel, LocalFunctionStartsWithEmptyTensorMap) {
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(18);
  OperatorSetIdProto *custom_os = model.add_opset_import();
  custom_os->set_domain("custom");
  custom_os->set_version(1);

  // Function "F" has a single formal input "x" and a node that references
  // a name ("leak") which exists in the caller's tensor map but is NOT a
  // declared function input. Because the function runs in an isolated
  // child RuntimeContext that starts empty (then gets only "x" bound),
  // dispatching the body must fail: "leak" cannot be resolved.
  FunctionProto *func = model.add_functions();
  func->set_name("F");
  func->set_domain("custom");
  func->add_input("x");
  func->add_output("out");
  NodeProto *fn = func->add_node();
  fn->set_op_type("Add");
  fn->add_input("x");
  fn->add_input("leak"); // not a function input -> must be invisible.
  fn->add_output("out");

  GraphProto *g = model.add_graph();
  g->set_name("main");
  NodeProto *call = g->add_node();
  call->set_op_type("F");
  call->set_domain("custom");
  call->add_input("x");
  call->add_output("y");

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("x", Tensor::FromFloat("x", {2}, {1.0f, 2.0f}));
  rt.Set("leak", Tensor::FromFloat("leak", {2}, {100.0f, 200.0f}));

  // The function body references "leak" which is not bound as a formal
  // input. With proper isolation the lookup throws.
  EXPECT_THROW(RunModelViaSession(model, rt), std::invalid_argument);

  // The caller's tensor map is untouched: "leak" is still there and the
  // function's formal output "out" did not leak in either.
  EXPECT_TRUE(rt.Has("leak"));
  EXPECT_FALSE(rt.Has("out"));
}

// Companion to the test above: the same model with "leak" passed as the
// function's formal input "x" succeeds, demonstrating that the function
// body sees only what was explicitly bound and that ``kernel_ctx()`` is
// shared (the Add kernel is dispatched against the caller's opset).
TEST(RunModel, LocalFunctionSharesKernelContextOnlyWithEmptyTensorMap) {
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(18);
  OperatorSetIdProto *custom_os = model.add_opset_import();
  custom_os->set_domain("custom");
  custom_os->set_version(1);

  FunctionProto *func = model.add_functions();
  func->set_name("Twice");
  func->set_domain("custom");
  func->add_input("v");
  func->add_output("out");
  NodeProto *fn = func->add_node();
  fn->set_op_type("Add");
  fn->add_input("v");
  fn->add_input("v");
  fn->add_output("internal_tmp"); // an intermediate inside the function.
  NodeProto *fn2 = func->add_node();
  fn2->set_op_type("Identity");
  fn2->add_input("internal_tmp");
  fn2->add_output("out");

  GraphProto *g = model.add_graph();
  g->set_name("main");
  NodeProto *call = g->add_node();
  call->set_op_type("Twice");
  call->set_domain("custom");
  call->add_input("a");
  call->add_output("y");

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("a", Tensor::FromFloat("a", {2}, {3.0f, 4.0f}));
  // Pre-populate the caller's tensor map with a name that collides with
  // a function-local intermediate. The function must not see or
  // overwrite this caller-side value; after the call the caller's
  // value must still be there.
  rt.Set("internal_tmp", Tensor::FromFloat("internal_tmp", {1}, {-1.0f}));

  RunModelViaSession(model, rt);

  ASSERT_TRUE(rt.Has("y"));
  EXPECT_FLOAT_EQ(rt.Get("y").AsFloat()[0], 6.0f);
  EXPECT_FLOAT_EQ(rt.Get("y").AsFloat()[1], 8.0f);
  // The function intermediate must not leak back to the caller: the
  // caller's pre-existing "internal_tmp" is preserved unchanged.
  ASSERT_TRUE(rt.Has("internal_tmp"));
  ASSERT_EQ(rt.Get("internal_tmp").shape, (std::vector<int64_t>{1}));
  EXPECT_FLOAT_EQ(rt.Get("internal_tmp").AsFloat()[0], -1.0f);
}

// A local subgraph (e.g. the body of an If/Loop/Scan node) must start
// with a *copy* of the caller's tensor map so it can reference outer-scope
// names. Intermediates produced inside the subgraph must NOT be propagated
// back to the caller's tensor map: only the values declared as subgraph
// outputs are visible to the caller (under the node's output names).
TEST(RunModel, LocalSubgraphCopiesCallerTensorsButHidesIntermediates) {
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(18);

  GraphProto *g = model.add_graph();
  g->set_name("main");
  NodeProto *if_node = g->add_node();
  if_node->set_op_type("If");
  if_node->add_input("cond");
  if_node->add_output("out");

  // ``then_branch`` references an outer-scope tensor ``outer`` (only
  // present in the caller's tensor map) via an Add node, and produces a
  // subgraph-local intermediate ``branch_tmp`` that must not leak back.
  AttributeProto *then_attr = if_node->add_attribute();
  then_attr->set_name("then_branch");
  then_attr->set_type(AttributeProto::AttributeType::GRAPH);
  GraphProto *then_g = then_attr->add_g();
  then_g->set_name("then_graph");
  NodeProto *then_add = then_g->add_node();
  then_add->set_op_type("Add");
  then_add->add_input("outer");
  then_add->add_input("outer");
  then_add->add_output("branch_tmp");
  NodeProto *then_id = then_g->add_node();
  then_id->set_op_type("Identity");
  then_id->add_input("branch_tmp");
  then_id->add_output("z");
  then_g->add_output()->set_name("z");

  // Minimal else_branch, never taken in this test.
  AttributeProto *else_attr = if_node->add_attribute();
  else_attr->set_name("else_branch");
  else_attr->set_type(AttributeProto::AttributeType::GRAPH);
  GraphProto *else_g = else_attr->add_g();
  else_g->set_name("else_graph");
  NodeProto *else_id = else_g->add_node();
  else_id->set_op_type("Identity");
  else_id->add_input("outer");
  else_id->add_output("z");
  else_g->add_output()->set_name("z");

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("cond", Tensor::FromBool("cond", {}, {1}));
  rt.Set("outer", Tensor::FromFloat("outer", {2}, {5.0f, 7.0f}));

  RunModelViaSession(model, rt);

  // The subgraph can see ``outer`` (caller's tensor map was copied into
  // the child) and produced the declared output.
  ASSERT_TRUE(rt.Has("out"));
  ASSERT_EQ(rt.Get("out").shape, (std::vector<int64_t>{2}));
  EXPECT_FLOAT_EQ(rt.Get("out").AsFloat()[0], 10.0f);
  EXPECT_FLOAT_EQ(rt.Get("out").AsFloat()[1], 14.0f);

  // The caller's pre-existing tensors are preserved.
  EXPECT_TRUE(rt.Has("cond"));
  EXPECT_TRUE(rt.Has("outer"));

  // Subgraph intermediates do NOT leak into the caller's tensor map.
  EXPECT_FALSE(rt.Has("branch_tmp"));
  // The subgraph's formal output name ``z`` (distinct from the node's
  // output name ``out``) likewise stays inside the child context.
  EXPECT_FALSE(rt.Has("z"));
}

// ---------------------------------------------------------------------------
// Custom kernel registration through RuntimeContext::RegisterCustomKernel.
// ---------------------------------------------------------------------------

// A custom kernel registered for an op outside the built-in dispatch table
// is invoked by RunNode; its output is written back to the RuntimeContext.
TEST(RunNodes, RunNodeDispatchesCustomKernelForUnknownOp) {
  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("x", Tensor::FromFloat("x", {3}, {1.0f, 2.0f, 3.0f}));

  int call_count = 0;
  rt.RegisterCustomKernel(
      "my.domain", "Scale", [&call_count](const NodeProto &node, RuntimeContext &ctx) {
        ++call_count;
        // Read the "factor" attribute (default 1.0f).
        float factor = 1.0f;
        for (int i = 0; i < node.attribute_size(); ++i) {
          const AttributeProto &a = node.attribute(i);
          if (a.name() == "factor") {
            factor = a.f();
          }
        }
        const Tensor &in = ctx.Get(node.input(0));
        std::vector<float> out(static_cast<size_t>(in.element_count()));
        const float *src = in.AsFloat();
        for (size_t i = 0; i < out.size(); ++i) {
          out[i] = src[i] * factor;
        }
        ctx.Put(node.output(0), Tensor::FromFloat(node.output(0), in.shape, out));
      });

  NodeProto node = MakeNode("Scale", {"x"}, {"y"}, "my.domain");
  AttributeProto *attr = node.add_attribute();
  attr->set_name("factor");
  attr->set_type(AttributeProto::AttributeType::FLOAT);
  attr->set_f(3.0f);

  RunNode(node, rt);
  EXPECT_EQ(call_count, 1);
  const Tensor &y = rt.tensors().at("y");
  ASSERT_EQ(y.element_count(), 3);
  const float *yp = y.AsFloat();
  EXPECT_FLOAT_EQ(yp[0], 3.0f);
  EXPECT_FLOAT_EQ(yp[1], 6.0f);
  EXPECT_FLOAT_EQ(yp[2], 9.0f);
}

// A custom kernel registered under the default ONNX domain overrides the
// built-in dispatch-table entry with the same key.
TEST(RunNodes, RunNodeCustomKernelOverridesBuiltin) {
  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("x", Tensor::FromFloat("x", {3}, {-1.0f, -2.0f, -3.0f}));

  // Replace Abs with negation: a custom override must take precedence over
  // the entry that KernelDispatchTable() would otherwise resolve.
  rt.RegisterCustomKernel("", "Abs", [](const NodeProto &node, RuntimeContext &ctx) {
    const Tensor &in = ctx.Get(node.input(0));
    std::vector<float> out(static_cast<size_t>(in.element_count()));
    const float *src = in.AsFloat();
    for (size_t i = 0; i < out.size(); ++i) {
      out[i] = -src[i];
    }
    ctx.Put(node.output(0), Tensor::FromFloat(node.output(0), in.shape, out));
  });

  NodeProto node = MakeNode("Abs", {"x"}, {"y"});
  RunNode(node, rt);

  const Tensor &y = rt.tensors().at("y");
  ASSERT_EQ(y.element_count(), 3);
  const float *yp = y.AsFloat();
  EXPECT_FLOAT_EQ(yp[0], 1.0f);
  EXPECT_FLOAT_EQ(yp[1], 2.0f);
  EXPECT_FLOAT_EQ(yp[2], 3.0f);
}

// Unregistering a custom kernel that overrode a built-in op restores the
// built-in original: the next RunNode dispatches to KernelDispatchTable()
// again. UnregisterCustomKernel returns true when an entry was removed.
TEST(RunNodes, UnregisterCustomKernelRestoresBuiltin) {
  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("x", Tensor::FromFloat("x", {3}, {-1.0f, -2.0f, -3.0f}));

  // Override Abs with negation, then verify the override is used.
  rt.RegisterCustomKernel("", "Abs", [](const NodeProto &node, RuntimeContext &ctx) {
    const Tensor &in = ctx.Get(node.input(0));
    std::vector<float> out(static_cast<size_t>(in.element_count()));
    const float *src = in.AsFloat();
    for (size_t i = 0; i < out.size(); ++i) {
      out[i] = -src[i];
    }
    ctx.Put(node.output(0), Tensor::FromFloat(node.output(0), in.shape, out));
  });

  NodeProto node = MakeNode("Abs", {"x"}, {"y"});
  RunNode(node, rt);
  {
    const Tensor &y = rt.tensors().at("y");
    ASSERT_EQ(y.element_count(), 3);
    const float *yp = y.AsFloat();
    EXPECT_FLOAT_EQ(yp[0], 1.0f);
    EXPECT_FLOAT_EQ(yp[1], 2.0f);
    EXPECT_FLOAT_EQ(yp[2], 3.0f);
  }

  // The empty domain is normalised to "ai.onnx", so unregistering with the
  // explicit "ai.onnx" domain removes the same entry.
  EXPECT_TRUE(rt.UnregisterCustomKernel("ai.onnx", "Abs"));

  // The built-in Abs is restored: negatives become their absolute value.
  RunNode(node, rt);
  {
    const Tensor &y = rt.tensors().at("y");
    ASSERT_EQ(y.element_count(), 3);
    const float *yp = y.AsFloat();
    EXPECT_FLOAT_EQ(yp[0], 1.0f);
    EXPECT_FLOAT_EQ(yp[1], 2.0f);
    EXPECT_FLOAT_EQ(yp[2], 3.0f);
  }

  // A second unregister for the same key finds nothing to remove.
  EXPECT_FALSE(rt.UnregisterCustomKernel("", "Abs"));
}

// Unregistering a custom-only op (one with no built-in dispatch entry) makes
// RunNode fail again with std::invalid_argument, matching the pre-registration
// behaviour.
TEST(RunNodes, UnregisterCustomKernelForUnknownOpFailsAgain) {
  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("x", Tensor::FromFloat("x", {3}, {1.0f, 2.0f, 3.0f}));

  rt.RegisterCustomKernel("my.domain", "Scale", [](const NodeProto &node, RuntimeContext &ctx) {
    const Tensor &in = ctx.Get(node.input(0));
    std::vector<float> out(static_cast<size_t>(in.element_count()));
    const float *src = in.AsFloat();
    for (size_t i = 0; i < out.size(); ++i) {
      out[i] = src[i] * src[i];
    }
    ctx.Put(node.output(0), Tensor::FromFloat(node.output(0), in.shape, out));
  });

  NodeProto node = MakeNode("Scale", {"x"}, {"y"}, "my.domain");
  RunNode(node, rt);
  EXPECT_EQ(rt.tensors().at("y").element_count(), 3);

  EXPECT_TRUE(rt.UnregisterCustomKernel("my.domain", "Scale"));
  EXPECT_THROW(RunNode(node, rt), std::invalid_argument);
}

// Running the model chains a built-in kernel and a custom kernel together;
// the CustomKernelMap survives across nodes within the same context.
TEST(RunModel, CustomKernelChainsWithBuiltinKernels) {
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(18);
  OperatorSetIdProto *custom_os = model.add_opset_import();
  custom_os->set_domain("my.domain");
  custom_os->set_version(1);

  GraphProto *g = model.add_graph();
  g->set_name("main");
  NodeProto *n1 = g->add_node();
  n1->set_op_type("Abs");
  n1->add_input("x");
  n1->add_output("a");
  NodeProto *n2 = g->add_node();
  n2->set_op_type("Scale");
  n2->set_domain("my.domain");
  n2->add_input("a");
  n2->add_output("y");
  AttributeProto *attr = n2->add_attribute();
  attr->set_name("factor");
  attr->set_type(AttributeProto::AttributeType::FLOAT);
  attr->set_f(2.0f);

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("x", Tensor::FromFloat("x", {3}, {-1.0f, -2.0f, -3.0f}));
  rt.RegisterCustomKernel("my.domain", "Scale", [](const NodeProto &node, RuntimeContext &ctx) {
    float factor = 1.0f;
    for (int i = 0; i < node.attribute_size(); ++i) {
      if (node.attribute(i).name() == "factor") {
        factor = node.attribute(i).f();
      }
    }
    const Tensor &in = ctx.Get(node.input(0));
    std::vector<float> out(static_cast<size_t>(in.element_count()));
    const float *src = in.AsFloat();
    for (size_t i = 0; i < out.size(); ++i) {
      out[i] = src[i] * factor;
    }
    ctx.Put(node.output(0), Tensor::FromFloat(node.output(0), in.shape, out));
  });

  RunModelViaSession(model, rt);
  const Tensor &y = rt.tensors().at("y");
  ASSERT_EQ(y.element_count(), 3);
  const float *yp = y.AsFloat();
  EXPECT_FLOAT_EQ(yp[0], 2.0f);
  EXPECT_FLOAT_EQ(yp[1], 4.0f);
  EXPECT_FLOAT_EQ(yp[2], 6.0f);
}

// Without a registered custom kernel, an unknown op fails as before.
TEST(RunNodes, RunNodeUnknownOpWithoutCustomKernelThrows) {
  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("x", Tensor::FromFloat("x", {3}, {1.0f, 2.0f, 3.0f}));
  NodeProto node = MakeNode("Scale", {"x"}, {"y"}, "my.domain");
  EXPECT_THROW(RunNode(node, rt), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Global (process-wide) custom kernel registration through
// core::runtime::RegisterGlobalCustomKernel.
// ---------------------------------------------------------------------------

// A globally registered custom kernel is picked up by RunNode on any
// RuntimeContext, without registering it on that context. Cleared afterwards
// so the registration does not leak into other tests.
TEST(RunNodes, RunNodeDispatchesGlobalCustomKernelForUnknownOp) {
  core::runtime::ClearGlobalCustomKernels();
  core::runtime::RegisterGlobalCustomKernel(
      "my.domain", "Scale", [](const NodeProto &node, RuntimeContext &ctx) {
        float factor = 1.0f;
        for (int i = 0; i < node.attribute_size(); ++i) {
          if (node.attribute(i).name() == "factor") {
            factor = node.attribute(i).f();
          }
        }
        const Tensor &in = ctx.Get(node.input(0));
        std::vector<float> out(static_cast<size_t>(in.element_count()));
        const float *src = in.AsFloat();
        for (size_t i = 0; i < out.size(); ++i) {
          out[i] = src[i] * factor;
        }
        ctx.Put(node.output(0), Tensor::FromFloat(node.output(0), in.shape, out));
      });

  NodeProto node = MakeNode("Scale", {"x"}, {"y"}, "my.domain");
  AttributeProto *attr = node.add_attribute();
  attr->set_name("factor");
  attr->set_type(AttributeProto::AttributeType::FLOAT);
  attr->set_f(3.0f);

  // A fresh context that never called RegisterCustomKernel still resolves the
  // globally registered kernel.
  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("x", Tensor::FromFloat("x", {3}, {1.0f, 2.0f, 3.0f}));
  RunNode(node, rt);
  const Tensor &y = rt.tensors().at("y");
  ASSERT_EQ(y.element_count(), 3);
  const float *yp = y.AsFloat();
  EXPECT_FLOAT_EQ(yp[0], 3.0f);
  EXPECT_FLOAT_EQ(yp[1], 6.0f);
  EXPECT_FLOAT_EQ(yp[2], 9.0f);

  // Unregistering the global kernel makes the unknown op fail again.
  EXPECT_TRUE(core::runtime::UnregisterGlobalCustomKernel("my.domain", "Scale"));
  EXPECT_FALSE(core::runtime::UnregisterGlobalCustomKernel("my.domain", "Scale"));
  EXPECT_THROW(RunNode(node, rt), std::invalid_argument);
}

// A per-context custom kernel overrides a global one for the same key, while
// the global kernel still overrides the built-in dispatch entry.
TEST(RunNodes, PerContextCustomKernelOverridesGlobalCustomKernel) {
  core::runtime::ClearGlobalCustomKernels();
  // Global override of Abs multiplies by 10 (distinct from both the built-in
  // Abs and the per-context override below).
  core::runtime::RegisterGlobalCustomKernel(
      "", "Abs", [](const NodeProto &node, RuntimeContext &ctx) {
        const Tensor &in = ctx.Get(node.input(0));
        std::vector<float> out(static_cast<size_t>(in.element_count()));
        const float *src = in.AsFloat();
        for (size_t i = 0; i < out.size(); ++i) {
          out[i] = src[i] * 10.0f;
        }
        ctx.Put(node.output(0), Tensor::FromFloat(node.output(0), in.shape, out));
      });

  NodeProto node = MakeNode("Abs", {"x"}, {"y"});

  // With only the global kernel registered, the global override applies.
  {
    RuntimeContext rt(KernelContext(DefaultOpset(18)));
    rt.Set("x", Tensor::FromFloat("x", {3}, {1.0f, 2.0f, 3.0f}));
    RunNode(node, rt);
    const float *yp = rt.tensors().at("y").AsFloat();
    EXPECT_FLOAT_EQ(yp[0], 10.0f);
    EXPECT_FLOAT_EQ(yp[1], 20.0f);
    EXPECT_FLOAT_EQ(yp[2], 30.0f);
  }

  // A per-context override (negation) takes precedence over the global one.
  {
    RuntimeContext rt(KernelContext(DefaultOpset(18)));
    rt.Set("x", Tensor::FromFloat("x", {3}, {1.0f, 2.0f, 3.0f}));
    rt.RegisterCustomKernel("", "Abs", [](const NodeProto &node, RuntimeContext &ctx) {
      const Tensor &in = ctx.Get(node.input(0));
      std::vector<float> out(static_cast<size_t>(in.element_count()));
      const float *src = in.AsFloat();
      for (size_t i = 0; i < out.size(); ++i) {
        out[i] = -src[i];
      }
      ctx.Put(node.output(0), Tensor::FromFloat(node.output(0), in.shape, out));
    });
    RunNode(node, rt);
    const float *yp = rt.tensors().at("y").AsFloat();
    EXPECT_FLOAT_EQ(yp[0], -1.0f);
    EXPECT_FLOAT_EQ(yp[1], -2.0f);
    EXPECT_FLOAT_EQ(yp[2], -3.0f);
  }

  core::runtime::ClearGlobalCustomKernels();
  EXPECT_TRUE(core::runtime::GlobalCustomKernels().empty());
}

// A node whose output is allocated from an allocator other than the session's
// unique allocator is rejected by RuntimeSession::Run's default output
// allocator check.
TEST(RunNodes, RuntimeSessionRejectsOutputFromForeignAllocator) {
  core::runtime::SimpleRawBufferAllocator session_alloc(8);
  core::runtime::SimpleRawBufferAllocator foreign_alloc(8);
  RuntimeContext rt(KernelContext(DefaultOpset(18)),
                    core::runtime::RuntimeContextOptions{.allocator = &session_alloc});
  rt.Set("x", Tensor::FromFloat("x", {3}, {1.0f, 2.0f, 3.0f}, &session_alloc));

  // Custom kernel produces its output from ``foreign_alloc`` rather than the
  // session's allocator.
  rt.RegisterCustomKernel(
      "my.domain", "Foreign", [&foreign_alloc](const NodeProto &node, RuntimeContext &ctx) {
        const Tensor &in = ctx.Get(node.input(0));
        std::vector<float> out(static_cast<size_t>(in.element_count()), 0.0f);
        ctx.Put(node.output(0), Tensor::FromFloat(node.output(0), in.shape, out, &foreign_alloc));
      });

  std::vector<NodeProto> nodes;
  nodes.push_back(MakeNode("Foreign", {"x"}, {"y"}, "my.domain"));
  utils::RepeatedProtoField<NodeProto> node_field(nodes);
  core::runtime::ExecutionPlan plan(node_field, {});
  core::runtime::RuntimeSession session(plan);
  EXPECT_THROW(session.Run(rt), std::invalid_argument);
}

// The same graph runs successfully when the session is built with
// ``allow_external_output_allocators`` enabled, letting a kernel return an
// output allocated outside the session's common allocator.
TEST(RunNodes, RuntimeSessionAllowsOutputFromForeignAllocatorWhenOptionSet) {
  core::runtime::SimpleRawBufferAllocator session_alloc(8);
  core::runtime::SimpleRawBufferAllocator foreign_alloc(8);
  RuntimeContext rt(KernelContext(DefaultOpset(18)),
                    core::runtime::RuntimeContextOptions{.allocator = &session_alloc});
  rt.Set("x", Tensor::FromFloat("x", {3}, {1.0f, 2.0f, 3.0f}, &session_alloc));

  rt.RegisterCustomKernel(
      "my.domain", "Foreign", [&foreign_alloc](const NodeProto &node, RuntimeContext &ctx) {
        const Tensor &in = ctx.Get(node.input(0));
        std::vector<float> out(static_cast<size_t>(in.element_count()));
        const float *src = in.AsFloat();
        for (size_t i = 0; i < out.size(); ++i) {
          out[i] = src[i] * 2.0f;
        }
        ctx.Put(node.output(0), Tensor::FromFloat(node.output(0), in.shape, out, &foreign_alloc));
      });

  std::vector<NodeProto> nodes;
  nodes.push_back(MakeNode("Foreign", {"x"}, {"y"}, "my.domain"));
  utils::RepeatedProtoField<NodeProto> node_field(nodes);
  core::runtime::ExecutionPlan plan(node_field, {});
  core::runtime::RuntimeSession session(
      plan, core::runtime::RuntimeSessionOptions{.allow_external_output_allocators = true});
  EXPECT_TRUE(session.allow_external_output_allocators());
  session.Run(rt);

  ASSERT_TRUE(rt.Has("y"));
  const Tensor &y = rt.Get("y");
  ASSERT_EQ(y.element_count(), 3);
  EXPECT_EQ(y.allocation_owner(), &foreign_alloc);
  const float *yp = y.AsFloat();
  EXPECT_FLOAT_EQ(yp[0], 2.0f);
  EXPECT_FLOAT_EQ(yp[1], 4.0f);
  EXPECT_FLOAT_EQ(yp[2], 6.0f);
}

// A RuntimeContext built with a dedicated I/O allocator (in addition to its
// execution allocator) routes the kernel invocation that produces a declared
// graph output through the I/O allocator, while an intermediate value that
// is not a declared output keeps allocating from the execution allocator.
// This exercises step 5 of the buffer-reuse arena plan (see
// docs/next_steps/2026-08_buffer_reuse_arena.rst): "route declared graph
// outputs directly to the I/O arena".
TEST(RunNodes, RuntimeSessionRoutesDeclaredOutputsToIOAllocator) {
  core::runtime::SimpleRawBufferAllocator execution_alloc(8);
  core::runtime::SimpleRawBufferAllocator io_alloc(8);
  RuntimeContext rt(KernelContext(DefaultOpset(18)),
                    core::runtime::RuntimeContextOptions{.allocator = &execution_alloc,
                                                         .io_allocator = &io_alloc});
  rt.Set("x", Tensor::FromFloat("x", {3}, {1.0f, 2.0f, 3.0f}, &execution_alloc));

  auto scale_by = [](float factor) {
    return [factor](const NodeProto &node, RuntimeContext &ctx) {
      const Tensor &in = ctx.Get(node.input(0));
      std::vector<float> out(static_cast<size_t>(in.element_count()));
      const float *src = in.AsFloat();
      for (size_t i = 0; i < out.size(); ++i) {
        out[i] = src[i] * factor;
      }
      ctx.Put(node.output(0), Tensor::FromFloat(node.output(0), in.shape, out, ctx.allocator()));
    };
  };
  rt.RegisterCustomKernel("my.domain", "Double", scale_by(2.0f));
  rt.RegisterCustomKernel("my.domain", "Increment", scale_by(1.0f));

  ModelProto model;
  model.set_ir_version(10);
  GraphProto *g = model.add_graph();
  g->set_name("main");
  // "z" is an intermediate, not a declared graph output.
  NodeProto *n1 = g->add_node();
  n1->set_op_type("Double");
  n1->set_domain("my.domain");
  n1->add_input("x");
  n1->add_output("z");
  // "y" is the graph's only declared output.
  NodeProto *n2 = g->add_node();
  n2->set_op_type("Increment");
  n2->set_domain("my.domain");
  n2->add_input("z");
  n2->add_output("y");
  g->add_output()->set_name("y");

  RuntimeSession session(model);
  session.Run(rt);

  ASSERT_TRUE(rt.Has("z"));
  ASSERT_TRUE(rt.Has("y"));
  const Tensor &z = rt.Get("z");
  const Tensor &y = rt.Get("y");
  EXPECT_EQ(z.allocation_owner(), &execution_alloc);
  EXPECT_EQ(y.allocation_owner(), &io_alloc);
  const float *yp = y.AsFloat();
  EXPECT_FLOAT_EQ(yp[0], 2.0f);
  EXPECT_FLOAT_EQ(yp[1], 4.0f);
  EXPECT_FLOAT_EQ(yp[2], 6.0f);
}

// Without a dedicated I/O allocator (the pre-existing single-allocator
// configuration), every output — declared or intermediate — keeps allocating
// from the execution allocator, preserving prior behaviour.
TEST(RunNodes, RuntimeSessionKeepsSingleAllocatorBehaviorWithoutIOAllocator) {
  core::runtime::SimpleRawBufferAllocator execution_alloc(8);
  RuntimeContext rt(KernelContext(DefaultOpset(18)),
                    core::runtime::RuntimeContextOptions{.allocator = &execution_alloc});
  rt.Set("x", Tensor::FromFloat("x", {3}, {1.0f, 2.0f, 3.0f}, &execution_alloc));

  rt.RegisterCustomKernel("my.domain", "Increment", [](const NodeProto &node, RuntimeContext &ctx) {
    const Tensor &in = ctx.Get(node.input(0));
    std::vector<float> out(static_cast<size_t>(in.element_count()));
    const float *src = in.AsFloat();
    for (size_t i = 0; i < out.size(); ++i) {
      out[i] = src[i] + 1.0f;
    }
    ctx.Put(node.output(0), Tensor::FromFloat(node.output(0), in.shape, out, ctx.allocator()));
  });

  ModelProto model;
  model.set_ir_version(10);
  GraphProto *g = model.add_graph();
  g->set_name("main");
  NodeProto *n = g->add_node();
  n->set_op_type("Increment");
  n->set_domain("my.domain");
  n->add_input("x");
  n->add_output("y");
  g->add_output()->set_name("y");

  RuntimeSession session(model);
  session.Run(rt);

  ASSERT_TRUE(rt.Has("y"));
  EXPECT_EQ(rt.Get("y").allocation_owner(), &execution_alloc);
}

// ---------------------------------------------------------------------------
// Release-unused-intermediates tests
// ---------------------------------------------------------------------------

TEST(RunNodes, CollectNodeInputsPlainNode) {
  NodeProto node = MakeNode("Add", {"x", "y"}, {"z"});
  auto inputs = core::runtime::RuntimeSession::CollectNodeInputs(node);
  EXPECT_EQ(inputs, (std::vector<std::string>{"x", "y"}));
}

TEST(RunNodes, CollectNodeInputsSkipsEmptyAndDedups) {
  NodeProto node = MakeNode("Add", {"x", "", "x"}, {"z"});
  auto inputs = core::runtime::RuntimeSession::CollectNodeInputs(node);
  EXPECT_EQ(inputs, (std::vector<std::string>{"x"}));
}

TEST(RunNodes, CollectNodeInputsIncludesSubgraphCaptures) {
  // ``If`` node with then/else subgraphs each capturing an outer name.
  NodeProto node;
  node.set_op_type("If");
  node.add_input("cond");

  auto add_subgraph = [](NodeProto &n, const std::string &attr_name,
                         const std::string &captured_name, const std::string &out_name) {
    AttributeProto attr;
    attr.set_name(attr_name);
    attr.set_type(AttributeProto::AttributeType::GRAPH);
    GraphProto &g = attr.ref_g();
    // Body: out_name = Identity(captured_name)
    NodeProto inner;
    inner.set_op_type("Identity");
    inner.add_input(captured_name);
    inner.add_output(out_name);
    g.ref_node().push_back(inner);
    ValueInfoProto vi;
    vi.set_name(out_name);
    g.ref_output().push_back(vi);
    n.ref_attribute().push_back(attr);
  };
  add_subgraph(node, "then_branch", "a", "y");
  add_subgraph(node, "else_branch", "b", "y");

  auto inputs = core::runtime::RuntimeSession::CollectNodeInputs(node);
  EXPECT_EQ(inputs, (std::vector<std::string>{"cond", "a", "b"}));
}

TEST(RunNodes, RunGraphReleaseIntermediatesRemovesUnusedAndEmitsEvent) {
  // y = Add(Abs(x), z) — after running, "t" (the intermediate) must be gone
  // from the context, "y" (declared output) must remain, and "x" / "z"
  // (graph inputs already in the context) must also remain.
  using core::runtime::RuntimeEventAction;

  GraphProto graph;
  ValueInfoProto vi_x;
  vi_x.set_name("x");
  ValueInfoProto vi_z;
  vi_z.set_name("z");
  ValueInfoProto vi_y;
  vi_y.set_name("y");
  graph.ref_input().push_back(vi_x);
  graph.ref_input().push_back(vi_z);
  graph.ref_output().push_back(vi_y);
  graph.ref_node().push_back(MakeNode("Abs", {"x"}, {"t"}));
  graph.ref_node().push_back(MakeNode("Add", {"t", "z"}, {"y"}));

  RuntimeContext rt(KernelContext(DefaultOpset(18)),
                    core::runtime::RuntimeContextOptions{.events_enabled = true});
  rt.set_release_intermediates(true);
  rt.Set("x", Tensor::FromFloat("x", {2}, {-1.0f, 2.0f}));
  rt.Set("z", Tensor::FromFloat("z", {2}, {10.0f, 20.0f}));

  RunGraphViaSession(graph, rt);

  // "t" was released, "y" / "x" / "z" survived.
  EXPECT_FALSE(rt.Has("t"));
  EXPECT_TRUE(rt.Has("y"));
  EXPECT_TRUE(rt.Has("x"));
  EXPECT_TRUE(rt.Has("z"));

  // At least one kRemove event was emitted for "t".
  bool saw_remove_t = false;
  for (const auto &ev : rt.events()) {
    if (ev.action == RuntimeEventAction::kRemove && ev.name == "t") {
      saw_remove_t = true;
      break;
    }
  }
  EXPECT_TRUE(saw_remove_t);

  // Default behaviour (release disabled) keeps the intermediate around so
  // callers can still fetch it after run.
  RuntimeContext rt2(KernelContext(DefaultOpset(18)));
  rt2.Set("x", Tensor::FromFloat("x", {2}, {-1.0f, 2.0f}));
  rt2.Set("z", Tensor::FromFloat("z", {2}, {10.0f, 20.0f}));
  RunGraphViaSession(graph, rt2);
  EXPECT_TRUE(rt2.Has("t"));
  EXPECT_TRUE(rt2.Has("y"));
}

TEST(RunNodes, ExecutionPlanIsCachedAcrossRunGraphInvocations) {
  // GetExecutionPlan returns the same instance on subsequent calls for
  // the same GraphProto, so the release analysis is paid only once.
  GraphProto graph;
  ValueInfoProto vi_x;
  vi_x.set_name("x");
  ValueInfoProto vi_y;
  vi_y.set_name("y");
  graph.ref_input().push_back(vi_x);
  graph.ref_output().push_back(vi_y);
  graph.ref_node().push_back(MakeNode("Abs", {"x"}, {"t"}));
  graph.ref_node().push_back(MakeNode("Neg", {"t"}, {"y"}));

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.set_release_intermediates(true);
  const core::runtime::ExecutionPlan &plan1 = rt.GetExecutionPlan(graph);
  const core::runtime::ExecutionPlan &plan2 = rt.GetExecutionPlan(graph);
  EXPECT_EQ(&plan1, &plan2);
  EXPECT_EQ(plan1.num_nodes(), 2u);
  // "x" / "y" are kept (declared input / output); "t" is an intermediate.
  EXPECT_TRUE(plan1.keep().count("x"));
  EXPECT_TRUE(plan1.keep().count("y"));

  // Two successive RunGraph calls both reuse the cached plan.
  rt.Set("x", Tensor::FromFloat("x", {2}, {-1.0f, 2.0f}));
  RunGraphViaSession(graph, rt);
  EXPECT_FALSE(rt.Has("t"));
  EXPECT_TRUE(rt.Has("y"));
  rt.Remove("y");
  rt.Put("x", Tensor::FromFloat("x", {2}, {-3.0f, 4.0f}));
  RunGraphViaSession(graph, rt);
  EXPECT_FALSE(rt.Has("t"));
  EXPECT_TRUE(rt.Has("y"));
  // Cached plan still the same instance after both runs.
  EXPECT_EQ(&rt.GetExecutionPlan(graph), &plan1);
}

TEST(RuntimeSession, InitializesKernelsThenRunsAndReleases) {
  // A RuntimeSession is constructed with a precomputed ExecutionPlan; its
  // kernels are resolved on the first Run against the supplied RuntimeContext
  // and only then is it run. Running it must produce the graph outputs and
  // release the scheduled intermediates, exactly like running a model's
  // graph or `SubgraphSession` (which build and run through a RuntimeSession
  // internally).
  using core::runtime::ExecutionPlan;
  using core::runtime::RuntimeSession;

  GraphProto graph;
  ValueInfoProto vi_x;
  vi_x.set_name("x");
  ValueInfoProto vi_z;
  vi_z.set_name("z");
  ValueInfoProto vi_y;
  vi_y.set_name("y");
  graph.ref_input().push_back(vi_x);
  graph.ref_input().push_back(vi_z);
  graph.ref_output().push_back(vi_y);
  graph.ref_node().push_back(MakeNode("Abs", {"x"}, {"t"}));
  graph.ref_node().push_back(MakeNode("Add", {"t", "z"}, {"y"}));

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.set_release_intermediates(true);
  rt.Set("x", Tensor::FromFloat("x", {2}, {-1.0f, 2.0f}));
  rt.Set("z", Tensor::FromFloat("z", {2}, {10.0f, 20.0f}));

  const ExecutionPlan &plan = rt.GetExecutionPlan(graph);
  RuntimeSession session(plan);
  session.Run(rt);

  // "t" was released, "y" (output) survived with the expected values.
  EXPECT_FALSE(rt.Has("t"));
  ASSERT_TRUE(rt.Has("y"));
  const float *y = rt.Get("y").AsFloat();
  ASSERT_EQ(rt.Get("y").element_count(), 2);
  EXPECT_FLOAT_EQ(y[0], 1.0f + 10.0f);
  EXPECT_FLOAT_EQ(y[1], 2.0f + 20.0f);
}

TEST(RuntimeSession, IsReusableAcrossMultipleRuns) {
  // The same session, built once (kernels initialized once), can be run again
  // with fresh inputs without re-resolving the kernels.
  using core::runtime::ExecutionPlan;
  using core::runtime::RuntimeSession;

  GraphProto graph;
  ValueInfoProto vi_x;
  vi_x.set_name("x");
  ValueInfoProto vi_y;
  vi_y.set_name("y");
  graph.ref_input().push_back(vi_x);
  graph.ref_output().push_back(vi_y);
  graph.ref_node().push_back(MakeNode("Abs", {"x"}, {"t"}));
  graph.ref_node().push_back(MakeNode("Neg", {"t"}, {"y"}));

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.set_release_intermediates(true);
  const ExecutionPlan &plan = rt.GetExecutionPlan(graph);
  RuntimeSession session(plan);

  rt.Set("x", Tensor::FromFloat("x", {2}, {-1.0f, 2.0f}));
  session.Run(rt);
  EXPECT_FALSE(rt.Has("t"));
  ASSERT_TRUE(rt.Has("y"));
  EXPECT_FLOAT_EQ(rt.Get("y").AsFloat()[0], -1.0f);
  EXPECT_FLOAT_EQ(rt.Get("y").AsFloat()[1], -2.0f);

  rt.Remove("y");
  rt.Put("x", Tensor::FromFloat("x", {2}, {-3.0f, 4.0f}));
  session.Run(rt);
  EXPECT_FALSE(rt.Has("t"));
  ASSERT_TRUE(rt.Has("y"));
  EXPECT_FLOAT_EQ(rt.Get("y").AsFloat()[0], -3.0f);
  EXPECT_FLOAT_EQ(rt.Get("y").AsFloat()[1], -4.0f);
}

TEST(RuntimeSession, ConstructsExactlyOneKernelPerNodeAcrossMultipleRuns) {
  // Registers a synthetic op whose NodeKernelFn factory (kernel construction)
  // and the returned kernel's Run (per-run execution) each bump their own
  // counter, so this test can assert directly: the factory — i.e. kernel
  // construction — runs exactly once per node no matter how many times the
  // session is run, while the invoke path runs once per node per Run() call.
  using core::runtime::NodeKernelFn;
  using core::runtime::RegisterKernelFn;

  int construct_count = 0;
  int invoke_count = 0;

  const std::string domain = "test.onnxlight.counting_kernel";
  RegisterKernelFn(domain, "CountingOp", core::symbolic::Device::kCPU,
                   [&construct_count, &invoke_count](const NodeProto &node, RuntimeContext &)
                       -> std::unique_ptr<core::runtime::KernelBase> {
                     ++construct_count;
                     return std::make_unique<TestLambdaKernel>(
                         node, [&invoke_count](const NodeProto &node, RuntimeContext &rt) {
                           ++invoke_count;
                           rt.Set(node.output(0), rt.Get(node.input(0)));
                         });
                   });

  GraphProto graph;
  ValueInfoProto vi_x;
  vi_x.set_name("x");
  ValueInfoProto vi_y;
  vi_y.set_name("y");
  graph.ref_input().push_back(vi_x);
  graph.ref_output().push_back(vi_y);
  graph.ref_node().push_back(MakeNode("CountingOp", {"x"}, {"y"}, domain));

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  const ExecutionPlan &plan = rt.GetExecutionPlan(graph);
  RuntimeSession session(plan);

  rt.Set("x", Tensor::FromFloat("x", {2}, {1.0f, 2.0f}));
  session.Run(rt);
  EXPECT_EQ(construct_count, 1);
  EXPECT_EQ(invoke_count, 1);

  rt.Remove("y");
  rt.Put("x", Tensor::FromFloat("x", {2}, {3.0f, 4.0f}));
  session.Run(rt);
  // The kernel is still constructed only once even on this second Run(), but
  // it is invoked again.
  EXPECT_EQ(construct_count, 1);
  EXPECT_EQ(invoke_count, 2);

  rt.Remove("y");
  rt.Put("x", Tensor::FromFloat("x", {2}, {5.0f, 6.0f}));
  session.Run(rt);
  EXPECT_EQ(construct_count, 1);
  EXPECT_EQ(invoke_count, 3);
}

TEST(RuntimeSession, KeepsResolvedKernelTuningImmutable) {
  using core::runtime::KernelTuningKey;
  using core::runtime::KernelTuningParameters;
  using core::runtime::KernelTuningSchema;
  using core::runtime::RegisterKernelFn;

  const std::string domain = "test.onnxlight.tunable_kernel";
  const KernelTuningKey key{
      "runtime_session_test", "TunableOp", "test", 0, core::symbolic::Device::kCPU, 1};
  KernelTuningParameters defaults{
      KernelTuningKey{key.library, key.kernel, key.implementation,
                      static_cast<int32_t>(TensorProto::DataType::FLOAT), key.device,
                      key.tuning_abi},
      {{"algorithm.threshold", int64_t{10}}}};
  core::runtime::RegisterKernelTuningSchema(KernelTuningSchema(defaults));
  RegisterKernelFn(
      domain, "TunableOp", core::symbolic::Device::kCPU,
      [key](const NodeProto &node, RuntimeContext &) -> std::unique_ptr<core::runtime::KernelBase> {
        return std::make_unique<TestTunableKernel>(node, key);
      });

  KernelTuningParameters first = defaults;
  first.values["algorithm.threshold"] = int64_t{20};
  core::runtime::GetKernelTuningRegistry().PublishProfiles(
      std::span<const KernelTuningParameters>(&first, 1));

  GraphProto graph;
  ValueInfoProto vi_x;
  vi_x.set_name("x");
  ValueInfoProto vi_y;
  vi_y.set_name("y");
  graph.ref_input().push_back(vi_x);
  graph.ref_output().push_back(vi_y);
  graph.ref_node().push_back(MakeNode("TunableOp", {"x"}, {"y"}, domain));

  RuntimeContext first_rt(KernelContext(DefaultOpset(18)));
  first_rt.Set("x", Tensor::FromFloat("x", {1}, {1.0f}));
  RuntimeSession first_session(first_rt.GetExecutionPlan(graph));
  core::runtime::KernelTuningRegistry &registry = core::runtime::GetKernelTuningRegistry();
  const core::runtime::KernelTuningRegistryAccessCounts before_first_run = registry.AccessCounts();
  first_session.Run(first_rt);
  const core::runtime::KernelTuningRegistryAccessCounts after_first_run = registry.AccessCounts();
  const uint64_t first_generation = first_session.tuning_generation();
  EXPECT_GT(first_generation, 0);
  EXPECT_EQ(first_rt.Get("y").AsInt64()[0], 20);
  EXPECT_EQ(after_first_run.snapshots - before_first_run.snapshots, 1u);
  EXPECT_EQ(after_first_run.lookups - before_first_run.lookups, 1u);
  EXPECT_EQ(after_first_run.resolutions - before_first_run.resolutions, 1u);
  const core::runtime::KernelTuningResolutionStatistics statistics =
      first_session.tuning_resolution_statistics();
  EXPECT_EQ(statistics.tunable_kernels, 1u);
  EXPECT_EQ(statistics.resolved_profiles, 1u);
  EXPECT_EQ(statistics.TotalDurationNs(),
            statistics.snapshot_duration_ns + statistics.resolution_duration_ns);

  KernelTuningParameters second = defaults;
  second.values["algorithm.threshold"] = int64_t{30};
  core::runtime::GetKernelTuningRegistry().PublishProfiles(
      std::span<const KernelTuningParameters>(&second, 1));

  first_rt.Remove("y");
  const core::runtime::KernelTuningRegistryAccessCounts before_hot_run = registry.AccessCounts();
  first_session.Run(first_rt);
  const core::runtime::KernelTuningRegistryAccessCounts after_hot_run = registry.AccessCounts();
  EXPECT_EQ(first_session.tuning_generation(), first_generation);
  EXPECT_EQ(first_rt.Get("y").AsInt64()[0], 20);
  EXPECT_EQ(after_hot_run, before_hot_run);
  EXPECT_EQ(first_session.tuning_resolution_statistics(), statistics);

  RuntimeContext second_rt(KernelContext(DefaultOpset(18)));
  second_rt.Set("x", Tensor::FromFloat("x", {1}, {1.0f}));
  RuntimeSession second_session(second_rt.GetExecutionPlan(graph));
  second_session.Run(second_rt);
  EXPECT_GT(second_session.tuning_generation(), first_generation);
  EXPECT_EQ(second_rt.Get("y").AsInt64()[0], 30);
}

TEST(RuntimeSession, ConstructsIfBranchKernelOnceAcrossMultipleRuns) {
  // Same guarantee as ConstructsExactlyOneKernelPerNodeAcrossMultipleRuns but
  // for a node nested inside an ``If`` branch: the branch's kernel must be
  // constructed once and reused across repeated Run() calls of the *outer*
  // session, not re-resolved every time the ``If`` node executes.
  using core::runtime::NodeKernelFn;
  using core::runtime::RegisterKernelFn;

  static int construct_count = 0;
  static int invoke_count = 0;
  construct_count = 0;
  invoke_count = 0;

  const std::string domain = "test.onnxlight.counting_kernel_if";
  RegisterKernelFn(
      domain, "CountingOp", core::symbolic::Device::kCPU,
      [](const NodeProto &node, RuntimeContext &) -> std::unique_ptr<core::runtime::KernelBase> {
        ++construct_count;
        return std::make_unique<TestLambdaKernel>(node,
                                                  [](const NodeProto &node, RuntimeContext &rt) {
                                                    ++invoke_count;
                                                    rt.Set(node.output(0), rt.Get(node.input(0)));
                                                  });
      });

  GraphProto then_branch;
  ValueInfoProto then_y;
  then_y.set_name("y");
  then_branch.ref_output().push_back(then_y);
  then_branch.ref_node().push_back(MakeNode("CountingOp", {"x"}, {"y"}, domain));

  GraphProto else_branch;
  ValueInfoProto else_y;
  else_y.set_name("y");
  else_branch.ref_output().push_back(else_y);
  else_branch.ref_node().push_back(MakeNode("Identity", {"x"}, {"y"}));

  NodeProto if_node = MakeNode("If", {"cond"}, {"y"});
  AttributeProto *then_attr = if_node.add_attribute();
  then_attr->set_name("then_branch");
  then_attr->set_type(AttributeProto::AttributeType::GRAPH);
  *then_attr->add_g() = then_branch;
  AttributeProto *else_attr = if_node.add_attribute();
  else_attr->set_name("else_branch");
  else_attr->set_type(AttributeProto::AttributeType::GRAPH);
  *else_attr->add_g() = else_branch;

  GraphProto graph;
  ValueInfoProto vi_cond;
  vi_cond.set_name("cond");
  ValueInfoProto vi_y;
  vi_y.set_name("y");
  graph.ref_input().push_back(vi_cond);
  graph.ref_output().push_back(vi_y);
  graph.ref_node().push_back(if_node);

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("x", Tensor::FromFloat("x", {2}, {1.0f, 2.0f}));
  rt.Set("cond", Tensor::FromBool("cond", {}, {true}));

  const ExecutionPlan &plan = rt.GetExecutionPlan(graph);
  RuntimeSession session(plan);

  session.Run(rt);
  EXPECT_EQ(construct_count, 1);
  EXPECT_EQ(invoke_count, 1);

  rt.Remove("y");
  session.Run(rt);
  // The If branch's kernel must still be constructed only once even though
  // the outer session (and hence the If node) has run a second time.
  EXPECT_EQ(construct_count, 1);
  EXPECT_EQ(invoke_count, 2);
}

TEST(RuntimeSession, ConstructsLoopBodyKernelOnceAcrossMultipleRuns) {
  // Same guarantee as ConstructsIfBranchKernelOnceAcrossMultipleRuns but for a
  // node nested inside a ``Loop`` body: the body's kernel must be constructed
  // once and reused both across iterations of a single Loop execution and
  // across repeated Run() calls of the *outer* session.
  using core::runtime::NodeKernelFn;
  using core::runtime::RegisterKernelFn;

  static int construct_count = 0;
  static int invoke_count = 0;
  construct_count = 0;
  invoke_count = 0;

  const std::string domain = "test.onnxlight.counting_kernel_loop";
  RegisterKernelFn(
      domain, "CountingOp", core::symbolic::Device::kCPU,
      [](const NodeProto &node, RuntimeContext &) -> std::unique_ptr<core::runtime::KernelBase> {
        ++construct_count;
        return std::make_unique<TestLambdaKernel>(node,
                                                  [](const NodeProto &node, RuntimeContext &rt) {
                                                    ++invoke_count;
                                                    rt.Set(node.output(0), rt.Get(node.input(0)));
                                                  });
      });

  GraphProto body;
  body.set_name("loop_body");
  body.add_input()->set_name("iter");
  body.add_input()->set_name("cond_in");
  body.add_input()->set_name("s_in");
  body.ref_node().push_back(MakeNode("CountingOp", {"s_in"}, {"s_out"}, domain));
  body.add_output()->set_name("cond_in");
  body.add_output()->set_name("s_out");

  NodeProto loop_node = MakeNode("Loop", {"M", "cond", "s_init"}, {"s_final"});
  AttributeProto *body_attr = loop_node.add_attribute();
  body_attr->set_name("body");
  body_attr->set_type(AttributeProto::AttributeType::GRAPH);
  *body_attr->add_g() = body;

  GraphProto graph;
  ValueInfoProto vi_m, vi_cond, vi_s_init, vi_s_final;
  vi_m.set_name("M");
  vi_cond.set_name("cond");
  vi_s_init.set_name("s_init");
  vi_s_final.set_name("s_final");
  graph.ref_input().push_back(vi_m);
  graph.ref_input().push_back(vi_cond);
  graph.ref_input().push_back(vi_s_init);
  graph.ref_output().push_back(vi_s_final);
  graph.ref_node().push_back(loop_node);

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("M", Tensor::FromInt64("M", {}, {3}));
  rt.Set("cond", Tensor::FromBool("cond", {}, {true}));
  rt.Set("s_init", Tensor::FromFloat("s_init", {}, {0.0f}));

  const ExecutionPlan &plan = rt.GetExecutionPlan(graph);
  RuntimeSession session(plan);

  session.Run(rt);
  // Constructed once for the whole first Run() (all 3 iterations reuse it),
  // but invoked once per iteration.
  EXPECT_EQ(construct_count, 1);
  EXPECT_EQ(invoke_count, 3);

  rt.Remove("s_final");
  rt.Put("s_init", Tensor::FromFloat("s_init", {}, {0.0f}));
  session.Run(rt);
  // Still constructed only once even though the outer session (and hence the
  // Loop node) has run a second time.
  EXPECT_EQ(construct_count, 1);
  EXPECT_EQ(invoke_count, 6);
}

TEST(RuntimeSession, ConstructsScanBodyKernelOnceAcrossMultipleRuns) {
  // Same guarantee as ConstructsLoopBodyKernelOnceAcrossMultipleRuns but for a
  // node nested inside a ``Scan`` body.
  using core::runtime::NodeKernelFn;
  using core::runtime::RegisterKernelFn;

  static int construct_count = 0;
  static int invoke_count = 0;
  construct_count = 0;
  invoke_count = 0;

  const std::string domain = "test.onnxlight.counting_kernel_scan";
  RegisterKernelFn(
      domain, "CountingOp", core::symbolic::Device::kCPU,
      [](const NodeProto &node, RuntimeContext &) -> std::unique_ptr<core::runtime::KernelBase> {
        ++construct_count;
        return std::make_unique<TestLambdaKernel>(node,
                                                  [](const NodeProto &node, RuntimeContext &rt) {
                                                    ++invoke_count;
                                                    rt.Set(node.output(0), rt.Get(node.input(0)));
                                                  });
      });

  GraphProto body;
  body.set_name("scan_body");
  body.add_input()->set_name("s_in");
  body.add_input()->set_name("x_in");
  body.ref_node().push_back(MakeNode("CountingOp", {"s_in"}, {"s_out"}, domain));
  body.add_output()->set_name("s_out");

  NodeProto scan_node = MakeNode("Scan", {"s_init", "x"}, {"s_final"});
  AttributeProto *body_attr = scan_node.add_attribute();
  body_attr->set_name("body");
  body_attr->set_type(AttributeProto::AttributeType::GRAPH);
  *body_attr->add_g() = body;
  AttributeProto *num_scan_inputs_attr = scan_node.add_attribute();
  num_scan_inputs_attr->set_name("num_scan_inputs");
  num_scan_inputs_attr->set_type(AttributeProto::AttributeType::INT);
  num_scan_inputs_attr->set_i(1);

  GraphProto graph;
  ValueInfoProto vi_s_init, vi_x, vi_s_final;
  vi_s_init.set_name("s_init");
  vi_x.set_name("x");
  vi_s_final.set_name("s_final");
  graph.ref_input().push_back(vi_s_init);
  graph.ref_input().push_back(vi_x);
  graph.ref_output().push_back(vi_s_final);
  graph.ref_node().push_back(scan_node);

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("s_init", Tensor::FromFloat("s_init", {}, {0.0f}));
  rt.Set("x", Tensor::FromFloat("x", {3}, {1.0f, 2.0f, 3.0f}));

  const ExecutionPlan &plan = rt.GetExecutionPlan(graph);
  RuntimeSession session(plan);

  session.Run(rt);
  EXPECT_EQ(construct_count, 1);
  EXPECT_EQ(invoke_count, 3);

  rt.Remove("s_final");
  rt.Put("s_init", Tensor::FromFloat("s_init", {}, {0.0f}));
  session.Run(rt);
  EXPECT_EQ(construct_count, 1);
  EXPECT_EQ(invoke_count, 6);
}

TEST(RuntimeSession, ConstructsSequenceMapBodyKernelOnceAcrossMultipleRuns) {
  // Same guarantee as ConstructsLoopBodyKernelOnceAcrossMultipleRuns but for a
  // node nested inside a ``SequenceMap`` body.
  using core::runtime::NodeKernelFn;
  using core::runtime::RegisterKernelFn;

  static int construct_count = 0;
  static int invoke_count = 0;
  construct_count = 0;
  invoke_count = 0;

  const std::string domain = "test.onnxlight.counting_kernel_seqmap";
  RegisterKernelFn(
      domain, "CountingOp", core::symbolic::Device::kCPU,
      [](const NodeProto &node, RuntimeContext &) -> std::unique_ptr<core::runtime::KernelBase> {
        ++construct_count;
        return std::make_unique<TestLambdaKernel>(node,
                                                  [](const NodeProto &node, RuntimeContext &rt) {
                                                    ++invoke_count;
                                                    rt.Set(node.output(0), rt.Get(node.input(0)));
                                                  });
      });

  GraphProto body;
  body.set_name("seq_map_body");
  body.add_input()->set_name("x_in");
  body.ref_node().push_back(MakeNode("CountingOp", {"x_in"}, {"y_out"}, domain));
  body.add_output()->set_name("y_out");

  NodeProto seq_map_node = MakeNode("SequenceMap", {"xs"}, {"ys"});
  AttributeProto *body_attr = seq_map_node.add_attribute();
  body_attr->set_name("body");
  body_attr->set_type(AttributeProto::AttributeType::GRAPH);
  *body_attr->add_g() = body;

  GraphProto graph;
  ValueInfoProto vi_xs, vi_ys;
  vi_xs.set_name("xs");
  vi_ys.set_name("ys");
  graph.ref_input().push_back(vi_xs);
  graph.ref_output().push_back(vi_ys);
  graph.ref_node().push_back(seq_map_node);

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.PutSequence("xs",
                 Sequence("xs", static_cast<int32_t>(DataType::FLOAT),
                          {Tensor::FromFloat("x0", {}, {1.0f}), Tensor::FromFloat("x1", {}, {2.0f}),
                           Tensor::FromFloat("x2", {}, {3.0f})}));

  const ExecutionPlan &plan = rt.GetExecutionPlan(graph);
  RuntimeSession session(plan);

  session.Run(rt);
  EXPECT_EQ(construct_count, 1);
  EXPECT_EQ(invoke_count, 3);

  rt.Remove("ys");
  session.Run(rt);
  EXPECT_EQ(construct_count, 1);
  EXPECT_EQ(invoke_count, 6);
}

TEST(SubgraphSession, ReusesInheritedKernelInitializationAcrossMultipleRuns) {
  using core::runtime::RegisterKernelFn;

  static int construct_count = 0;
  static int invoke_count = 0;
  construct_count = 0;
  invoke_count = 0;

  const std::string domain = "test.onnxlight.counting_kernel_subgraph";
  RegisterKernelFn(
      domain, "CountingOp", core::symbolic::Device::kCPU,
      [](const NodeProto &node, RuntimeContext &) -> std::unique_ptr<core::runtime::KernelBase> {
        ++construct_count;
        return std::make_unique<TestLambdaKernel>(node,
                                                  [](const NodeProto &node, RuntimeContext &rt) {
                                                    ++invoke_count;
                                                    rt.Set(node.output(0), rt.Get(node.input(0)));
                                                  });
      });

  GraphProto graph;
  graph.set_name("subgraph");
  graph.add_input()->set_name("x");
  graph.ref_node().push_back(MakeNode("CountingOp", {"x"}, {"y"}, domain));
  graph.add_output()->set_name("y");

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  core::runtime::SubgraphSession session(rt, graph);

  core::runtime::Tensors outputs =
      session.Run({{"x", Tensor::FromFloat("x", {2}, {1.0f, 2.0f})}}, rt, "body");
  ASSERT_EQ(outputs.size(), 1U);
  EXPECT_EQ(session.required_inputs(), (std::vector<std::string>{"x"}));
  EXPECT_EQ(construct_count, 1);
  EXPECT_EQ(invoke_count, 1);

  outputs = session.Run({{"x", Tensor::FromFloat("x", {2}, {3.0f, 4.0f})}}, rt, "body");
  ASSERT_EQ(outputs.size(), 1U);
  EXPECT_EQ(construct_count, 1);
  EXPECT_EQ(invoke_count, 2);
}

TEST(RuntimeSession, RejectsUnsupportedOpDuringKernelInitialization) {
  // Kernel resolution happens on the first Run (kernel initialization), so an
  // unsupported operator is rejected the first time the session is run.
  using core::runtime::ExecutionPlan;
  using core::runtime::RuntimeSession;

  GraphProto graph;
  ValueInfoProto vi_x;
  vi_x.set_name("x");
  ValueInfoProto vi_y;
  vi_y.set_name("y");
  graph.ref_input().push_back(vi_x);
  graph.ref_output().push_back(vi_y);
  graph.ref_node().push_back(MakeNode("ThisOpDoesNotExist", {"x"}, {"y"}));

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.set_release_intermediates(true);
  const ExecutionPlan &plan = rt.GetExecutionPlan(graph);
  RuntimeSession session(plan);
  EXPECT_THROW(session.Run(rt), std::invalid_argument);
}

TEST(RuntimeSession, RecordsRequiredInputsAndRejectsMissingOne) {
  // Kernel initialization records the external inputs the scheduled nodes read
  // (required_inputs), and Run verifies the RuntimeContext supplies every one
  // of them before executing any kernel.
  using core::runtime::ExecutionPlan;
  using core::runtime::RuntimeSession;

  GraphProto graph;
  ValueInfoProto vi_x;
  vi_x.set_name("x");
  ValueInfoProto vi_z;
  vi_z.set_name("z");
  ValueInfoProto vi_y;
  vi_y.set_name("y");
  graph.ref_input().push_back(vi_x);
  graph.ref_input().push_back(vi_z);
  graph.ref_output().push_back(vi_y);
  graph.ref_node().push_back(MakeNode("Abs", {"x"}, {"t"}));
  graph.ref_node().push_back(MakeNode("Add", {"t", "z"}, {"y"}));

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.set_release_intermediates(true);
  // Only "x" is provided; the plan also requires "z", so Run must reject it
  // before running the kernels.
  rt.Set("x", Tensor::FromFloat("x", {2}, {-1.0f, 2.0f}));

  const ExecutionPlan &plan = rt.GetExecutionPlan(graph);
  RuntimeSession session(plan);
  EXPECT_THROW(session.Run(rt), std::invalid_argument);

  // After the first (failed) Run the required inputs are known: "x" and "z"
  // (the intermediate "t" is produced by a node and is not required).
  const std::vector<std::string> &required = session.required_inputs();
  ASSERT_EQ(required.size(), 2u);
  EXPECT_EQ(required[0], "x");
  EXPECT_EQ(required[1], "z");

  // Supplying the missing input lets the session run to completion.
  rt.Set("z", Tensor::FromFloat("z", {2}, {10.0f, 20.0f}));
  session.Run(rt);
  ASSERT_TRUE(rt.Has("y"));
  EXPECT_FLOAT_EQ(rt.Get("y").AsFloat()[0], 1.0f + 10.0f);
  EXPECT_FLOAT_EQ(rt.Get("y").AsFloat()[1], 2.0f + 20.0f);
}

TEST(RunNodes, ExecutionPlanBuildsActions) {
  // BuildActions is metadata-driven: it locks the input on first use, allocates
  // a result (or reuses one in place) per output, executes each node, then frees
  // intermediates and unlocks inputs based on the in-place / lifetime
  // annotations carried by each node's metadata_props.
  GraphProto graph;
  ValueInfoProto vi_x;
  vi_x.set_name("x");
  ValueInfoProto vi_y;
  vi_y.set_name("y");
  graph.ref_input().push_back(vi_x);
  graph.ref_output().push_back(vi_y);
  graph.ref_node().push_back(MakeNode("Abs", {"x"}, {"t"}));
  graph.ref_node().push_back(MakeNode("Neg", {"t"}, {"y"}));

  // Node 0 (Abs) reuses input "x" in place for its output "t"; node 1 (Neg)
  // releases the intermediate "t" (its last use) and marks "x" as not used
  // after it.
  (*graph.mutable_node())[0].add_metadata(core::compute::kInPlaceReuseMetadataKey, "0:0:equal");
  (*graph.mutable_node())[1].add_metadata(core::compute::kReleaseAfterMetadataKey, "t");
  (*graph.mutable_node())[1].add_metadata(core::compute::kNotUsedAfterMetadataKey, "x");

  core::runtime::ExecutionPlan plan(graph);

  const std::vector<core::runtime::ExecuteAction> &actions = plan.actions();
  ASSERT_FALSE(actions.empty());
  using core::runtime::ExecuteActionKind;
  // The input is locked on first use (before the Abs node) and unlocked on last
  // use (after the Neg node).
  EXPECT_EQ(actions.front().kind(), ExecuteActionKind::kLockInput);
  EXPECT_EQ(actions.front().name(), "x");
  EXPECT_EQ(actions.back().kind(), ExecuteActionKind::kUnlockInput);
  EXPECT_EQ(actions.back().name(), "x");

  // Exactly one ExecuteNode action per node, in order.
  std::vector<size_t> execute_indices;
  bool t_deleted = false;
  bool t_inplace = false;
  bool y_allocated = false;
  for (const auto &action : actions) {
    if (action.kind() == ExecuteActionKind::kExecuteNode) {
      execute_indices.push_back(action.node_index());
    }
    if (action.kind() == ExecuteActionKind::kAllocateBuffer && action.name() == "t") {
      // "t" is produced in place from input "x": no fresh allocation.
      t_inplace = action.is_inplace();
      EXPECT_EQ(action.target(), "x");
      EXPECT_EQ(action.inplace().output_index, 0);
      EXPECT_EQ(action.inplace().input_index, 0);
    }
    if (action.kind() == ExecuteActionKind::kAllocateBuffer && action.name() == "y") {
      // "y" has no reuse opportunity: it is freshly allocated.
      y_allocated = true;
      EXPECT_FALSE(action.is_inplace());
    }
    if (action.kind() == ExecuteActionKind::kDeleteBuffer && action.name() == "t") {
      t_deleted = true;
    }
  }
  ASSERT_EQ(execute_indices.size(), 2u);
  EXPECT_EQ(execute_indices[0], 0u);
  EXPECT_EQ(execute_indices[1], 1u);
  EXPECT_TRUE(t_inplace);
  EXPECT_TRUE(y_allocated);
  EXPECT_TRUE(t_deleted);
}

TEST(RunNodes, ExecuteActionSummary) {
  using core::runtime::ExecuteAction;
  using core::runtime::ExecuteActionKind;

  // Node execution mentions the node index only.
  ExecuteAction execute(ExecuteActionKind::kExecuteNode, "", /*node_index=*/3);
  EXPECT_EQ(execute.summary(), "ExecuteNode node_index=3");

  // Lock / unlock and shape actions mention only the name.
  ExecuteAction lock(ExecuteActionKind::kLockInput, "x");
  EXPECT_EQ(lock.summary(), "LockInput name='x'");
  ExecuteAction create_shape(ExecuteActionKind::kCreateShape, "s");
  EXPECT_EQ(create_shape.summary(), "CreateShape name='s'");

  // A fresh buffer allocation mentions name and size.
  ExecuteAction alloc(ExecuteActionKind::kAllocateBuffer, "y", /*node_index=*/0,
                      /*size=*/16);
  EXPECT_EQ(alloc.summary(), "AllocateBuffer name='y' size=16");

  // An in-place allocation mentions the reuse decision and reused buffer.
  ExecuteAction inplace(ExecuteActionKind::kAllocateBuffer, "t", /*node_index=*/0,
                        /*size=*/8, /*target=*/"x", core::compute::InPlaceReuse{0, 0});
  EXPECT_EQ(inplace.summary(),
            "AllocateBuffer name='t' size=8 inplace(output=0, input=0) reuses='x'");

  // A transfer mentions both endpoints.
  ExecuteAction transfer(ExecuteActionKind::kTransfer, "a", /*node_index=*/0,
                         /*size=*/0, /*target=*/"b");
  EXPECT_EQ(transfer.summary(), "Transfer name='a' -> target='b'");

  // Sequence / map deletions mention only the name and expose stable names.
  ExecuteAction delete_sequence(ExecuteActionKind::kDeleteSequence, "seq");
  EXPECT_EQ(delete_sequence.kind_name(), std::string("DeleteSequence"));
  EXPECT_EQ(delete_sequence.summary(), "DeleteSequence name='seq'");
  ExecuteAction delete_map(ExecuteActionKind::kDeleteMap, "m");
  EXPECT_EQ(delete_map.kind_name(), std::string("DeleteMap"));
  EXPECT_EQ(delete_map.summary(), "DeleteMap name='m'");
}

TEST(RunNodes, ExecutionPlanShapeTagActions) {
  // A value tagged "shape" gets its shape created / destroyed (no data buffer),
  // while a regular result gets its buffer allocated / freed.
  GraphProto graph;
  ValueInfoProto vi_x;
  vi_x.set_name("x");
  ValueInfoProto vi_y;
  vi_y.set_name("y");
  graph.ref_input().push_back(vi_x);
  graph.ref_output().push_back(vi_y);
  graph.ref_node().push_back(MakeNode("Shape", {"x"}, {"s"}));
  graph.ref_node().push_back(MakeNode("Reshape", {"x", "s"}, {"y"}));

  // "s" is released at node 1 and is value-tagged as a shape; the input "x"
  // reaches its last use at node 1 and is unlocked there.
  (*graph.mutable_node())[1].add_metadata(core::compute::kReleaseAfterMetadataKey, "s");
  (*graph.mutable_node())[1].add_metadata(core::compute::kReleaseAfterShapeTagMetadataKey, "s");
  (*graph.mutable_node())[1].add_metadata(core::compute::kNotUsedAfterMetadataKey, "x");

  core::runtime::ExecutionPlan plan(graph);

  using core::runtime::ExecuteActionKind;
  bool s_shape_created = false;
  bool s_shape_deleted = false;
  bool s_buffer_touched = false;
  for (const auto &action : plan.actions()) {
    if (action.name() != "s") {
      continue;
    }
    if (action.kind() == ExecuteActionKind::kCreateShape) {
      s_shape_created = true;
    }
    if (action.kind() == ExecuteActionKind::kDeleteShape) {
      s_shape_deleted = true;
    }
    if (action.kind() == ExecuteActionKind::kAllocateBuffer ||
        action.kind() == ExecuteActionKind::kDeleteBuffer) {
      s_buffer_touched = true;
    }
  }
  EXPECT_TRUE(s_shape_created);
  EXPECT_TRUE(s_shape_deleted);
  EXPECT_FALSE(s_buffer_touched);
}

TEST(RunNodes, ExecutionPlanPeakMemoryActions) {
  // A node carrying a peak-memory estimate gets a temporary/scratch buffer
  // allocated right before it runs and deleted right after, sized from the
  // peak-memory metadata written by WritePeakMemoryToMetadata.
  GraphProto graph;
  ValueInfoProto vi_x;
  vi_x.set_name("x");
  ValueInfoProto vi_y;
  vi_y.set_name("y");
  graph.ref_input().push_back(vi_x);
  graph.ref_output().push_back(vi_y);
  graph.ref_node().push_back(MakeNode("Abs", {"x"}, {"t"}));
  graph.ref_node().push_back(MakeNode("Neg", {"t"}, {"y"}));

  (*graph.mutable_node())[1].add_metadata(core::compute::kReleaseAfterMetadataKey, "t");
  // Only node 0 needs a scratch buffer of 256 bytes.
  (*graph.mutable_node())[0].add_metadata(core::compute::kNodePeakMemoryMetadataKey, "256");

  core::runtime::ExecutionPlan plan(graph);

  using core::runtime::ExecuteActionKind;
  const std::vector<core::runtime::ExecuteAction> &actions = plan.actions();

  // Locate the temporary-buffer actions and the executions they wrap.
  std::optional<size_t> alloc_temp_index;
  std::optional<size_t> delete_temp_index;
  std::optional<size_t> execute_node0_index;
  size_t temp_action_count = 0;
  for (size_t i = 0; i < actions.size(); ++i) {
    const core::runtime::ExecuteAction &action = actions[i];
    if (action.kind() == ExecuteActionKind::kAllocateTemporaryBuffer) {
      ++temp_action_count;
      alloc_temp_index = i;
      EXPECT_EQ(action.node_index(), 0u);
      EXPECT_EQ(action.size(), 256u);
    }
    if (action.kind() == ExecuteActionKind::kDeleteTemporaryBuffer) {
      ++temp_action_count;
      delete_temp_index = i;
      EXPECT_EQ(action.node_index(), 0u);
      EXPECT_EQ(action.size(), 256u);
    }
    if (action.kind() == ExecuteActionKind::kExecuteNode && action.node_index() == 0u) {
      execute_node0_index = i;
    }
  }
  // Exactly one allocate + one delete, and only for the annotated node.
  EXPECT_EQ(temp_action_count, 2u);
  ASSERT_TRUE(alloc_temp_index.has_value());
  ASSERT_TRUE(delete_temp_index.has_value());
  ASSERT_TRUE(execute_node0_index.has_value());
  // The scratch buffer wraps the node execution: allocate before, delete after.
  EXPECT_LT(*alloc_temp_index, *execute_node0_index);
  EXPECT_LT(*execute_node0_index, *delete_temp_index);
}

// ---------------------------------------------------------------------------
// SubgraphEventGraphName tests
// Verify that events produced inside subgraphs carry the correct
// subgraph_node_index and subgraph_attr_name.
// ---------------------------------------------------------------------------

// Helper: build a minimal Loop body that adds a scalar ``one`` initializer to
// ``s_in`` and writes it to ``s_out``, while forwarding the loop condition.
static void FillLoopBody(GraphProto &body) {
  body.set_name("loop_body");
  body.add_input()->set_name("iter");
  body.add_input()->set_name("cond_in");
  body.add_input()->set_name("s_in");
  TensorProto *one = body.add_initializer();
  one->set_name("one");
  one->set_data_type(TensorProto::DataType::FLOAT);
  one->add_float_data(1.0f);
  NodeProto *add = body.add_node();
  add->set_op_type("Add");
  add->add_input("s_in");
  add->add_input("one");
  add->add_output("s_out");
  body.add_output()->set_name("cond_in");
  body.add_output()->set_name("s_out");
  body.add_output()->set_name("s_out");
}

TEST(SubgraphEventGraphName, LoopSubgraphEventsCarryBodyGraphName) {
  using core::runtime::RuntimeEventAction;

  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(18);

  GraphProto *g = model.add_graph();
  g->set_name("main");
  NodeProto *loop = g->add_node();
  loop->set_op_type("Loop");
  loop->add_input("M");
  loop->add_input("cond");
  loop->add_input("s_init");
  loop->add_output("s_final");
  loop->add_output("scan");

  AttributeProto *body_attr = loop->add_attribute();
  body_attr->set_name("body");
  body_attr->set_type(AttributeProto::AttributeType::GRAPH);
  FillLoopBody(*body_attr->add_g());

  RuntimeContext rt(KernelContext(DefaultOpset(18)),
                    core::runtime::RuntimeContextOptions{.events_enabled = true});
  rt.Set("M", Tensor::FromInt64("M", {}, {2}));
  rt.Set("cond", Tensor::FromBool("cond", {}, {1}));
  rt.Set("s_init", Tensor::FromFloat("s_init", {}, {0.0f}));
  rt.ClearEvents();

  RunModelViaSession(model, rt);

  // All events from the loop body subgraph must be tagged with
  // subgraph_attr_name "body". Events from the outer graph must have an
  // empty subgraph_attr_name.
  bool found_body_event = false;
  for (const auto &ev : rt.events()) {
    if (ev.subgraph_attr_name == "body") {
      found_body_event = true;
      break;
    }
  }
  EXPECT_TRUE(found_body_event) << "No event with subgraph_attr_name='body' found";

  // Top-level events must have an empty subgraph_attr_name.
  for (const auto &ev : rt.events()) {
    if (ev.subgraph_attr_name.empty()) {
      // At least one outer-graph event should exist (output tensors).
      break;
    }
  }
}

TEST(SubgraphEventGraphName, IfSubgraphEventsCarryBranchGraphName) {
  using core::runtime::RuntimeEventAction;

  // Build a trivial If model: cond -> If(then_branch, else_branch).
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(18);

  GraphProto *g = model.add_graph();
  g->set_name("main");
  NodeProto *if_node = g->add_node();
  if_node->set_op_type("If");
  if_node->add_input("cond");
  if_node->add_output("z");

  AttributeProto *then_attr = if_node->add_attribute();
  then_attr->set_name("then_branch");
  then_attr->set_type(AttributeProto::AttributeType::GRAPH);
  FillConstantBranch(*then_attr->add_g(), "then_g", "t", "z", 1.0f);

  AttributeProto *else_attr = if_node->add_attribute();
  else_attr->set_name("else_branch");
  else_attr->set_type(AttributeProto::AttributeType::GRAPH);
  FillConstantBranch(*else_attr->add_g(), "else_g", "e", "z", 2.0f);

  // Run with cond = true: the then_branch executes.
  RuntimeContext rt_true(KernelContext(DefaultOpset(18)),
                         core::runtime::RuntimeContextOptions{.events_enabled = true});
  rt_true.Set("cond", Tensor::FromBool("cond", {}, {1}));
  rt_true.ClearEvents();
  RunModelViaSession(model, rt_true);

  bool found_then = false;
  for (const auto &ev : rt_true.events()) {
    if (ev.subgraph_attr_name == "then_branch") {
      found_then = true;
      break;
    }
  }
  EXPECT_TRUE(found_then) << "No event with subgraph_attr_name='then_branch' found";

  // Run with cond = false: the else_branch executes.
  RuntimeContext rt_false(KernelContext(DefaultOpset(18)),
                          core::runtime::RuntimeContextOptions{.events_enabled = true});
  rt_false.Set("cond", Tensor::FromBool("cond", {}, {0}));
  rt_false.ClearEvents();
  RunModelViaSession(model, rt_false);

  bool found_else = false;
  for (const auto &ev : rt_false.events()) {
    if (ev.subgraph_attr_name == "else_branch") {
      found_else = true;
      break;
    }
  }
  EXPECT_TRUE(found_else) << "No event with subgraph_attr_name='else_branch' found";
}

TEST(SubgraphEventGraphName, ScanSubgraphEventsCarryBodyGraphName) {
  using core::runtime::RuntimeEventAction;

  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(18);

  GraphProto *g = model.add_graph();
  g->set_name("main");
  NodeProto *scan = g->add_node();
  scan->set_op_type("Scan");
  scan->add_input("state0");
  scan->add_input("x");
  scan->add_output("state_final");
  scan->add_output("y");
  AttributeProto *num_attr = scan->add_attribute();
  num_attr->set_name("num_scan_inputs");
  num_attr->set_type(AttributeProto::AttributeType::INT);
  num_attr->set_i(1);

  AttributeProto *body_attr = scan->add_attribute();
  body_attr->set_name("body");
  body_attr->set_type(AttributeProto::AttributeType::GRAPH);
  GraphProto *body = body_attr->add_g();
  body->set_name("scan_body");
  body->add_input()->set_name("state_in");
  body->add_input()->set_name("x_in");
  NodeProto *add = body->add_node();
  add->set_op_type("Add");
  add->add_input("state_in");
  add->add_input("x_in");
  add->add_output("state_out");
  body->add_output()->set_name("state_out");
  body->add_output()->set_name("state_out");

  RuntimeContext rt(KernelContext(DefaultOpset(18)),
                    core::runtime::RuntimeContextOptions{.events_enabled = true});
  rt.Set("state0", Tensor::FromFloat("state0", {}, {0.0f}));
  rt.Set("x", Tensor::FromFloat("x", {3}, {1.0f, 2.0f, 3.0f}));
  rt.ClearEvents();

  RunModelViaSession(model, rt);

  bool found_body = false;
  for (const auto &ev : rt.events()) {
    if (ev.subgraph_attr_name == "body") {
      found_body = true;
      break;
    }
  }
  EXPECT_TRUE(found_body) << "No event with subgraph_attr_name='body' found";
}

TEST(SubgraphEventGraphName, TopLevelEventsHaveEmptyGraphName) {
  // A plain two-node graph (no subgraphs) must produce only events
  // with an empty subgraph_attr_name.
  using core::runtime::RuntimeEventAction;

  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(18);

  GraphProto *g = model.add_graph();
  g->set_name("main");
  *g->add_node() = MakeNode("Abs", {"x"}, {"t"});
  *g->add_node() = MakeNode("Neg", {"t"}, {"y"});

  RuntimeContext rt(KernelContext(DefaultOpset(18)),
                    core::runtime::RuntimeContextOptions{.events_enabled = true});
  rt.Set("x", Tensor::FromFloat("x", {2}, {-1.0f, 2.0f}));
  rt.ClearEvents();

  RunModelViaSession(model, rt);

  for (const auto &ev : rt.events()) {
    EXPECT_TRUE(ev.subgraph_attr_name.empty())
        << "Expected empty subgraph_attr_name for top-level event, got: " << ev.subgraph_attr_name;
  }
}

TEST(NodeHelpers, GetAttributeShapeOrDefaultReturnsShape) {
  NodeProto node;
  node.set_op_type("Pool");
  AttributeProto *attr = node.add_attribute();
  attr->set_name("kernel_shape");
  attr->set_type(AttributeProto::AttributeType::INTS);
  attr->add_ints(3);
  attr->add_ints(3);

  const core::runtime::Shape result =
      core::runtime::GetAttributeShapeOrDefault(node, "kernel_shape", core::runtime::Shape{});
  ASSERT_EQ(result.size(), static_cast<size_t>(2));
  EXPECT_EQ(result[0], 3);
  EXPECT_EQ(result[1], 3);
}

TEST(NodeHelpers, GetAttributeShapeOrDefaultReturnsFallback) {
  NodeProto node;
  node.set_op_type("Pool");

  const core::runtime::Shape fallback{1, 1};
  const core::runtime::Shape result =
      core::runtime::GetAttributeShapeOrDefault(node, "kernel_shape", fallback);
  ASSERT_EQ(result.size(), static_cast<size_t>(2));
  EXPECT_EQ(result[0], 1);
  EXPECT_EQ(result[1], 1);
}

// ---------------------------------------------------------------------------
// node_helpers.h exception paths
// ---------------------------------------------------------------------------

TEST(NodeHelpers, GetInputEmptyNameThrows) {
  NodeProto node = MakeNode("Abs", {""}, {"y"});
  TensorMap tensors;
  EXPECT_THROW(core::runtime::GetInput(node, 0, tensors), std::invalid_argument);
}

TEST(NodeHelpers, GetInputMissingFromMapThrows) {
  NodeProto node = MakeNode("Abs", {"x"}, {"y"});
  TensorMap tensors;
  EXPECT_THROW(core::runtime::GetInput(node, 0, tensors), std::invalid_argument);
}

TEST(NodeHelpers, GetInputReturnsTensor) {
  NodeProto node = MakeNode("Abs", {"x"}, {"y"});
  TensorMap tensors;
  tensors["x"] = Tensor::FromFloat("x", {1}, {2.0f});
  const Tensor &t = core::runtime::GetInput(node, 0, tensors);
  EXPECT_FLOAT_EQ(t.AsFloat()[0], 2.0f);
}

TEST(NodeHelpers, GetOptionalInputReturnsNullWhenSlotAbsent) {
  NodeProto node = MakeNode("Abs", {"x"}, {"y"});
  TensorMap tensors;
  EXPECT_EQ(core::runtime::GetOptionalInput(node, 5, tensors), nullptr);
}

TEST(NodeHelpers, GetOptionalInputReturnsNullWhenNameEmpty) {
  NodeProto node = MakeNode("Abs", {""}, {"y"});
  TensorMap tensors;
  EXPECT_EQ(core::runtime::GetOptionalInput(node, 0, tensors), nullptr);
}

TEST(NodeHelpers, GetOptionalInputMissingFromMapThrows) {
  NodeProto node = MakeNode("Abs", {"x"}, {"y"});
  TensorMap tensors;
  EXPECT_THROW(core::runtime::GetOptionalInput(node, 0, tensors), std::invalid_argument);
}

TEST(NodeHelpers, GetOptionalInputReturnsTensor) {
  NodeProto node = MakeNode("Abs", {"x"}, {"y"});
  TensorMap tensors;
  tensors["x"] = Tensor::FromFloat("x", {1}, {3.0f});
  const Tensor *t = core::runtime::GetOptionalInput(node, 0, tensors);
  ASSERT_NE(t, nullptr);
  EXPECT_FLOAT_EQ(t->AsFloat()[0], 3.0f);
}

TEST(NodeHelpers, SetOutputEmptyNameThrows) {
  NodeProto node = MakeNode("Abs", {"x"}, {""});
  TensorMap tensors;
  EXPECT_THROW(core::runtime::SetOutput(node, 0, Tensor::FromFloat("y", {1}, {1.0f}), tensors),
               std::invalid_argument);
}

TEST(NodeHelpers, SetOutputStoresTensor) {
  NodeProto node = MakeNode("Abs", {"x"}, {"y"});
  TensorMap tensors;
  core::runtime::SetOutput(node, 0, Tensor::FromFloat("tmp", {1}, {4.0f}), tensors);
  ASSERT_TRUE(tensors.count("y"));
  EXPECT_EQ(tensors["y"].name, "y");
  EXPECT_FLOAT_EQ(tensors["y"].AsFloat()[0], 4.0f);
}

TEST(NodeHelpers, SetOutputContextEmptyNameThrows) {
  NodeProto node = MakeNode("Abs", {"x"}, {""});
  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  EXPECT_THROW(core::runtime::SetOutput(node, 0, Tensor::FromFloat("y", {1}, {1.0f}), rt),
               std::invalid_argument);
}

TEST(NodeHelpers, SetOutputContextStoresTensor) {
  NodeProto node = MakeNode("Abs", {"x"}, {"y"});
  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  core::runtime::SetOutput(node, 0, Tensor::FromFloat("tmp", {1}, {5.0f}), rt);
  ASSERT_TRUE(rt.Has("y"));
  EXPECT_FLOAT_EQ(rt.Get("y").AsFloat()[0], 5.0f);
}

TEST(NodeHelpers, GetInputSequenceEmptyNameThrows) {
  NodeProto node = MakeNode("SequenceAt", {""}, {"y"});
  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  EXPECT_THROW(core::runtime::GetInputSequence(node, 0, rt), std::invalid_argument);
}

TEST(NodeHelpers, GetInputSequenceMissingThrows) {
  NodeProto node = MakeNode("SequenceAt", {"seq"}, {"y"});
  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  EXPECT_THROW(core::runtime::GetInputSequence(node, 0, rt), std::invalid_argument);
}

TEST(NodeHelpers, GetInputSequenceReturnsSequence) {
  NodeProto node = MakeNode("SequenceAt", {"seq"}, {"y"});
  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  Sequence seq;
  seq.name = "seq";
  seq.values.push_back(Tensor::FromFloat("e0", {1}, {6.0f}));
  rt.PutSequence("seq", std::move(seq));
  const Sequence &got = core::runtime::GetInputSequence(node, 0, rt);
  ASSERT_EQ(got.size(), static_cast<size_t>(1));
  EXPECT_FLOAT_EQ(got.at(0).AsFloat()[0], 6.0f);
}

TEST(NodeHelpers, SetOutputSequenceEmptyNameThrows) {
  NodeProto node = MakeNode("SequenceEmpty", {}, {""});
  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  EXPECT_THROW(core::runtime::SetOutputSequence(node, 0, Sequence{}, rt), std::invalid_argument);
}

TEST(NodeHelpers, SetOutputSequenceStoresSequence) {
  NodeProto node = MakeNode("SequenceEmpty", {}, {"seq"});
  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  core::runtime::SetOutputSequence(node, 0, Sequence{}, rt);
  EXPECT_TRUE(rt.HasSequence("seq"));
}

TEST(NodeHelpers, RequireInputCountThrowsOnMismatch) {
  NodeProto node = MakeNode("Add", {"x"}, {"y"});
  EXPECT_THROW(core::runtime::RequireInputCount(node, 2), std::invalid_argument);
  EXPECT_NO_THROW(core::runtime::RequireInputCount(node, 1));
}

TEST(NodeHelpers, RequireMinInputCountThrowsWhenTooFew) {
  NodeProto node = MakeNode("Add", {"x"}, {"y"});
  EXPECT_THROW(core::runtime::RequireMinInputCount(node, 2), std::invalid_argument);
  EXPECT_NO_THROW(core::runtime::RequireMinInputCount(node, 1));
}

TEST(NodeHelpers, RequireOutputCountThrowsOnMismatch) {
  NodeProto node = MakeNode("Add", {"a", "b"}, {"y"});
  EXPECT_THROW(core::runtime::RequireOutputCount(node, 2), std::invalid_argument);
  EXPECT_NO_THROW(core::runtime::RequireOutputCount(node, 1));
}

TEST(NodeHelpers, FindAttributeReturnsNullWhenMissing) {
  NodeProto node = MakeNode("Abs", {"x"}, {"y"});
  EXPECT_EQ(core::runtime::FindAttribute(node, "nope"), nullptr);
}

TEST(NodeHelpers, GetRequiredGraphAttributeMissingThrows) {
  NodeProto node = MakeNode("If", {"cond"}, {"y"});
  EXPECT_THROW(core::runtime::GetRequiredGraphAttribute(node, "then_branch"),
               std::invalid_argument);
}

TEST(NodeHelpers, GetRequiredGraphAttributeWrongTypeThrows) {
  NodeProto node = MakeNode("If", {"cond"}, {"y"});
  AttributeProto *attr = node.add_attribute();
  attr->set_name("then_branch");
  attr->set_type(AttributeProto::AttributeType::INT);
  attr->set_i(1);
  EXPECT_THROW(core::runtime::GetRequiredGraphAttribute(node, "then_branch"),
               std::invalid_argument);
}

TEST(NodeHelpers, GetRequiredGraphAttributeReturnsGraph) {
  NodeProto node = MakeNode("If", {"cond"}, {"y"});
  AttributeProto *attr = node.add_attribute();
  attr->set_name("then_branch");
  attr->set_type(AttributeProto::AttributeType::GRAPH);
  attr->ref_g().set_name("body");
  const GraphProto &g = core::runtime::GetRequiredGraphAttribute(node, "then_branch");
  EXPECT_EQ(g.name(), "body");
}

TEST(NodeHelpers, GetAttributeIntOrDefaultReturnsFallback) {
  NodeProto node = MakeNode("Op", {}, {});
  EXPECT_EQ(core::runtime::GetAttributeIntOrDefault(node, "axis", 7), 7);
}

TEST(NodeHelpers, GetAttributeIntOrDefaultWrongTypeThrows) {
  NodeProto node = MakeNode("Op", {}, {});
  AttributeProto *attr = node.add_attribute();
  attr->set_name("axis");
  attr->set_type(AttributeProto::AttributeType::FLOAT);
  attr->set_f(1.0f);
  EXPECT_THROW(core::runtime::GetAttributeIntOrDefault(node, "axis", 0), std::invalid_argument);
}

TEST(NodeHelpers, GetAttributeIntsOrDefaultWrongTypeThrows) {
  NodeProto node = MakeNode("Op", {}, {});
  AttributeProto *attr = node.add_attribute();
  attr->set_name("axes");
  attr->set_type(AttributeProto::AttributeType::INT);
  attr->set_i(1);
  EXPECT_THROW(core::runtime::GetAttributeIntsOrDefault(node, "axes", {}), std::invalid_argument);
}

TEST(NodeHelpers, GetAttributeShapeOrDefaultWrongTypeThrows) {
  NodeProto node = MakeNode("Op", {}, {});
  AttributeProto *attr = node.add_attribute();
  attr->set_name("kernel_shape");
  attr->set_type(AttributeProto::AttributeType::INT);
  attr->set_i(1);
  EXPECT_THROW(
      core::runtime::GetAttributeShapeOrDefault(node, "kernel_shape", core::runtime::Shape{}),
      std::invalid_argument);
}

TEST(NodeHelpers, GetAttributeFloatsOrDefaultWrongTypeThrows) {
  NodeProto node = MakeNode("Op", {}, {});
  AttributeProto *attr = node.add_attribute();
  attr->set_name("scales");
  attr->set_type(AttributeProto::AttributeType::INTS);
  attr->add_ints(1);
  EXPECT_THROW(core::runtime::GetAttributeFloatsOrDefault(node, "scales", {}),
               std::invalid_argument);
}

TEST(NodeHelpers, GetAttributeStringsOrDefaultWrongTypeThrows) {
  NodeProto node = MakeNode("Op", {}, {});
  AttributeProto *attr = node.add_attribute();
  attr->set_name("mode");
  attr->set_type(AttributeProto::AttributeType::STRING);
  attr->set_s("x");
  EXPECT_THROW(core::runtime::GetAttributeStringsOrDefault(node, "mode", {}),
               std::invalid_argument);
}

TEST(NodeHelpers, GetAttributeFloatOrDefaultWrongTypeThrows) {
  NodeProto node = MakeNode("Op", {}, {});
  AttributeProto *attr = node.add_attribute();
  attr->set_name("alpha");
  attr->set_type(AttributeProto::AttributeType::INT);
  attr->set_i(1);
  EXPECT_THROW(core::runtime::GetAttributeFloatOrDefault(node, "alpha", 0.0f),
               std::invalid_argument);
}

TEST(NodeHelpers, GetAttributeStringOrDefaultWrongTypeThrows) {
  NodeProto node = MakeNode("Op", {}, {});
  AttributeProto *attr = node.add_attribute();
  attr->set_name("mode");
  attr->set_type(AttributeProto::AttributeType::INT);
  attr->set_i(1);
  EXPECT_THROW(core::runtime::GetAttributeStringOrDefault(node, "mode", ""), std::invalid_argument);
}

TEST(NodeHelpers, GetRequiredAttributeStringMissingThrows) {
  NodeProto node = MakeNode("Op", {}, {});
  EXPECT_THROW(core::runtime::GetRequiredAttributeString(node, "mode"), std::invalid_argument);
}

TEST(NodeHelpers, GetRequiredAttributeStringWrongTypeThrows) {
  NodeProto node = MakeNode("Op", {}, {});
  AttributeProto *attr = node.add_attribute();
  attr->set_name("mode");
  attr->set_type(AttributeProto::AttributeType::INT);
  attr->set_i(1);
  EXPECT_THROW(core::runtime::GetRequiredAttributeString(node, "mode"), std::invalid_argument);
}

TEST(NodeHelpers, GetRequiredAttributeStringReturnsValue) {
  NodeProto node = MakeNode("Op", {}, {});
  AttributeProto *attr = node.add_attribute();
  attr->set_name("mode");
  attr->set_type(AttributeProto::AttributeType::STRING);
  attr->set_s("linear");
  EXPECT_EQ(core::runtime::GetRequiredAttributeString(node, "mode"), "linear");
}

TEST(NodeHelpers, GetRequiredAttributeIntMissingThrows) {
  NodeProto node = MakeNode("Op", {}, {});
  EXPECT_THROW(core::runtime::GetRequiredAttributeInt(node, "axis"), std::invalid_argument);
}

TEST(NodeHelpers, GetRequiredAttributeIntWrongTypeThrows) {
  NodeProto node = MakeNode("Op", {}, {});
  AttributeProto *attr = node.add_attribute();
  attr->set_name("axis");
  attr->set_type(AttributeProto::AttributeType::STRING);
  attr->set_s("x");
  EXPECT_THROW(core::runtime::GetRequiredAttributeInt(node, "axis"), std::invalid_argument);
}

TEST(NodeHelpers, GetRequiredAttributeIntReturnsValue) {
  NodeProto node = MakeNode("Op", {}, {});
  AttributeProto *attr = node.add_attribute();
  attr->set_name("axis");
  attr->set_type(AttributeProto::AttributeType::INT);
  attr->set_i(3);
  EXPECT_EQ(core::runtime::GetRequiredAttributeInt(node, "axis"), 3);
}

// ---------------------------------------------------------------------------
// RuntimeSession concrete-vs-symbolic shape validation (constructor option)
// ---------------------------------------------------------------------------

namespace {

// Adds a FLOAT tensor-typed value_info named ``name`` to ``vi`` with one
// ``Dimension`` per entry in ``dims``: a symbolic dimension when the string is
// non-empty, a concrete ``dim_value`` when the string is empty and the integer
// is ``>= 0``, and an unset (unknown) dimension when the integer is negative.
void AddFloatTensorValueInfo(ValueInfoProto *vi, const std::string &name,
                             const std::vector<std::pair<int64_t, std::string>> &dims) {
  vi->set_name(name);
  TypeProto::Tensor *tt = vi->ref_type().add_tensor_type();
  tt->set_elem_type(static_cast<int>(TensorProto::DataType::FLOAT));
  TensorShapeProto *shape = tt->add_shape();
  for (const auto &d : dims) {
    TensorShapeProto::Dimension *dim = shape->add_dim();
    if (!d.second.empty()) {
      dim->set_dim_param(d.second);
    } else if (d.first >= 0) {
      dim->set_dim_value(d.first);
    }
  }
}

// Builds a single-node ``Add`` model whose inputs (``x``, ``y``) and output
// (``z``) carry the supplied declared shapes.
ModelProto MakeAddModelWithShapes(const std::vector<std::pair<int64_t, std::string>> &x_shape,
                                  const std::vector<std::pair<int64_t, std::string>> &y_shape,
                                  const std::vector<std::pair<int64_t, std::string>> &z_shape) {
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(18);
  GraphProto *g = model.add_graph();
  g->set_name("check_shapes");
  NodeProto *node = g->add_node();
  node->set_op_type("Add");
  node->add_input("x");
  node->add_input("y");
  node->add_output("z");
  AddFloatTensorValueInfo(g->add_input(), "x", x_shape);
  AddFloatTensorValueInfo(g->add_input(), "y", y_shape);
  AddFloatTensorValueInfo(g->add_output(), "z", z_shape);
  return model;
}

} // namespace

TEST(RuntimeSessionCheckShapes, DefaultsToDisabled) {
  ModelProto model = MakeAddModelWithShapes({{2, ""}}, {{2, ""}}, {{2, ""}});
  RuntimeSession session(model);
  EXPECT_FALSE(session.check_shapes());
  RuntimeSession enabled(model, RuntimeSessionOptions{
                                    .parameters = RuntimeParameters(),
                                    .verbose = 0,
                                    .check_shapes = true,
                                });
  EXPECT_TRUE(enabled.check_shapes());
}

TEST(RuntimeSessionCheckShapes, PassesWhenConcreteMatchesSymbolic) {
  // x: [N, 3], y: [N, 3], z: [N, 3]; running with N == 2 keeps the symbolic
  // dimension consistent across inputs and output.
  ModelProto model =
      MakeAddModelWithShapes({{-1, "N"}, {3, ""}}, {{-1, "N"}, {3, ""}}, {{-1, "N"}, {3, ""}});
  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("x", Tensor::FromFloat("x", {2, 3}, {1, 2, 3, 4, 5, 6}));
  rt.Set("y", Tensor::FromFloat("y", {2, 3}, {6, 5, 4, 3, 2, 1}));
  RuntimeSession session(model, RuntimeSessionOptions{
                                    .parameters = RuntimeParameters(),
                                    .verbose = 0,
                                    .check_shapes = true,
                                });
  EXPECT_NO_THROW(session.Run(rt));
  ASSERT_TRUE(rt.Has("z"));
  EXPECT_EQ(rt.Get("z").shape, std::vector<int64_t>({2, 3}));
}

TEST(RuntimeSessionCheckShapes, RejectsConcreteDimMismatchOnOutput) {
  // z is declared FLOAT[N, 4] but the produced tensor is [2, 3]; the concrete
  // dimension 4 mismatch must be reported.
  ModelProto model =
      MakeAddModelWithShapes({{-1, "N"}, {3, ""}}, {{-1, "N"}, {3, ""}}, {{-1, "N"}, {4, ""}});
  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("x", Tensor::FromFloat("x", {2, 3}, {1, 2, 3, 4, 5, 6}));
  rt.Set("y", Tensor::FromFloat("y", {2, 3}, {6, 5, 4, 3, 2, 1}));
  RuntimeSession session(model, RuntimeSessionOptions{
                                    .parameters = RuntimeParameters(),
                                    .verbose = 0,
                                    .check_shapes = true,
                                });
  EXPECT_THROW(session.Run(rt), std::invalid_argument);
}

TEST(RuntimeSessionCheckShapes, RejectsInconsistentSymbolicBinding) {
  // Both inputs share the symbolic dimension ``N`` but are seeded with
  // different concrete values ([2] vs [3]); the inconsistency is caught during
  // the up-front input validation, before the Add kernel runs.
  ModelProto model = MakeAddModelWithShapes({{-1, "N"}}, {{-1, "N"}}, {{-1, "N"}});
  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("x", Tensor::FromFloat("x", {2}, {1, 2}));
  rt.Set("y", Tensor::FromFloat("y", {3}, {1, 2, 3}));
  RuntimeSession session(model, RuntimeSessionOptions{
                                    .parameters = RuntimeParameters(),
                                    .verbose = 0,
                                    .check_shapes = true,
                                });
  EXPECT_THROW(session.Run(rt), std::invalid_argument);
}

TEST(RuntimeSessionCheckShapes, DisabledFlagSkipsValidation) {
  // z declared FLOAT[N, 4] but produced as [2, 3]; with checking disabled the
  // mismatch is ignored and Run succeeds.
  ModelProto model =
      MakeAddModelWithShapes({{-1, "N"}, {3, ""}}, {{-1, "N"}, {3, ""}}, {{-1, "N"}, {4, ""}});
  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("x", Tensor::FromFloat("x", {2, 3}, {1, 2, 3, 4, 5, 6}));
  rt.Set("y", Tensor::FromFloat("y", {2, 3}, {6, 5, 4, 3, 2, 1}));
  RuntimeSession session(model);
  ASSERT_FALSE(session.check_shapes());
  EXPECT_NO_THROW(session.Run(rt));
}

TEST(RuntimeSessionCheckShapes, RejectsRankMismatch) {
  // z declared rank 1 (FLOAT[N]) but produced rank 2 ([2, 3]).
  ModelProto model =
      MakeAddModelWithShapes({{-1, "N"}, {3, ""}}, {{-1, "N"}, {3, ""}}, {{-1, "N"}});
  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("x", Tensor::FromFloat("x", {2, 3}, {1, 2, 3, 4, 5, 6}));
  rt.Set("y", Tensor::FromFloat("y", {2, 3}, {6, 5, 4, 3, 2, 1}));
  RuntimeSession session(model, RuntimeSessionOptions{
                                    .parameters = RuntimeParameters(),
                                    .verbose = 0,
                                    .check_shapes = true,
                                });
  EXPECT_THROW(session.Run(rt), std::invalid_argument);
}

TEST(RuntimeSessionCheckShapes, SetDeclaredShapesOnPlanBuiltSession) {
  // A session built from an ExecutionPlan alone carries no declared shapes
  // until SetDeclaredShapes is called; afterwards the check applies.
  ModelProto model =
      MakeAddModelWithShapes({{-1, "N"}, {3, ""}}, {{-1, "N"}, {3, ""}}, {{-1, "N"}, {4, ""}});
  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("x", Tensor::FromFloat("x", {2, 3}, {1, 2, 3, 4, 5, 6}));
  rt.Set("y", Tensor::FromFloat("y", {2, 3}, {6, 5, 4, 3, 2, 1}));
  const ExecutionPlan &plan = rt.GetExecutionPlan(model.ref_graph());
  RuntimeSession session(plan, RuntimeSessionOptions{
                                   .parameters = RuntimeParameters(),
                                   .verbose = 0,
                                   .check_shapes = true,
                               });
  // No declared shapes yet: nothing to validate, so the [2,3] output passes.
  EXPECT_NO_THROW(session.Run(rt));

  RuntimeContext rt2(KernelContext(DefaultOpset(18)));
  rt2.Set("x", Tensor::FromFloat("x", {2, 3}, {1, 2, 3, 4, 5, 6}));
  rt2.Set("y", Tensor::FromFloat("y", {2, 3}, {6, 5, 4, 3, 2, 1}));
  RuntimeSession session2(plan, RuntimeSessionOptions{
                                    .parameters = RuntimeParameters(),
                                    .verbose = 0,
                                    .check_shapes = true,
                                });
  session2.SetDeclaredShapes(model.ref_graph());
  EXPECT_THROW(session2.Run(rt2), std::invalid_argument);
}

// A ``Constant`` whose value tensor stores ``raw_data`` yields a borrowed
// runtime tensor viewing into the model proto's ``raw_data``. When that
// borrowed tensor is a graph output, ``RuntimeSession`` (built from the model
// so it knows the declared outputs) must detach it into an owned buffer before
// ``Run`` returns, so the output survives the model being released.
TEST(RunNodes, RuntimeSessionMaterializesBorrowedConstantOutput) {
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_domain("");
  os->set_version(18);
  GraphProto *graph = model.add_graph();
  graph->set_name("main");
  graph->add_output()->set_name("Y");

  NodeProto *node = graph->add_node();
  node->set_op_type("Constant");
  node->add_output("Y");
  AttributeProto *attr = node->add_attribute();
  attr->set_name("value");
  attr->set_type(AttributeProto::AttributeType::TENSOR);
  TensorProto *t = attr->add_t();
  t->set_data_type(TensorProto::DataType::FLOAT);
  t->add_dims(3);
  // Store the value in ``raw_data`` (not the typed ``float_data`` field) so the
  // Constant kernel borrows into the proto instead of owning the bytes.
  const float values[3] = {1.0f, 2.0f, 3.0f};
  std::vector<uint8_t> raw(sizeof(values));
  std::memcpy(raw.data(), values, sizeof(values));
  t->set_raw_data(utils::ByteSpan(raw));

  // Driving the model's own plan directly (no declared outputs) leaves the
  // output borrowing into the proto: this is the state the session must fix.
  {
    RuntimeContext rt(KernelContext(DefaultOpset(18)));
    const ExecutionPlan &plan = rt.GetExecutionPlan(model.ref_graph());
    RuntimeSession session(plan);
    session.Run(rt);
    ASSERT_TRUE(rt.Has("Y"));
    EXPECT_TRUE(rt.Get("Y").is_borrowed());
  }

  // Building the session from the ModelProto records the graph outputs, so Run
  // detaches the borrowed output into an owned tensor.
  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  RuntimeSession session(model);
  session.Run(rt);
  ASSERT_TRUE(rt.Has("Y"));
  const Tensor &y = rt.Get("Y");
  EXPECT_FALSE(y.is_borrowed());
  ASSERT_EQ(y.element_count(), 3);
  const float *got = y.AsFloat();
  EXPECT_FLOAT_EQ(got[0], 1.0f);
  EXPECT_FLOAT_EQ(got[1], 2.0f);
  EXPECT_FLOAT_EQ(got[2], 3.0f);
}

// A Scan node whose scan output carries an empty name must be skipped when
// its results are propagated back to the caller (PropagateOutputsToCaller).
TEST(RunModel, ScanNodeEmptyScanOutputNameIsSkipped) {
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(18);

  GraphProto *g = model.add_graph();
  g->set_name("main");
  NodeProto *scan = g->add_node();
  scan->set_op_type("Scan");
  scan->add_input("state0");
  scan->add_input("x");
  scan->add_output("state_final");
  // The scan output is intentionally unnamed: PropagateOutputsToCaller must
  // skip it without materialising a tensor.
  scan->add_output("");
  AttributeProto *num_attr = scan->add_attribute();
  num_attr->set_name("num_scan_inputs");
  num_attr->set_type(AttributeProto::AttributeType::INT);
  num_attr->set_i(1);

  AttributeProto *body_attr = scan->add_attribute();
  body_attr->set_name("body");
  body_attr->set_type(AttributeProto::AttributeType::GRAPH);
  GraphProto *body = body_attr->add_g();
  body->set_name("scan_body");
  body->add_input()->set_name("state_in");
  body->add_input()->set_name("x_in");
  NodeProto *add = body->add_node();
  add->set_op_type("Add");
  add->add_input("state_in");
  add->add_input("x_in");
  add->add_output("state_out");
  body->add_output()->set_name("state_out");
  body->add_output()->set_name("state_out");

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("state0", Tensor::FromFloat("state0", {}, {0.0f}));
  rt.Set("x", Tensor::FromFloat("x", {3}, {1.0f, 2.0f, 3.0f}));

  RunModelViaSession(model, rt);

  ASSERT_TRUE(rt.Has("state_final"));
  EXPECT_FLOAT_EQ(rt.Get("state_final").AsFloat()[0], 6.0f);
  // No tensor is created for the empty scan-output name.
  EXPECT_FALSE(rt.Has(""));
}

// An If node whose (only) output carries an empty name must be skipped by
// RunIfNode without producing a tensor for the caller.
TEST(RunModel, IfNodeEmptyOutputNameIsSkipped) {
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(18);

  GraphProto *g = model.add_graph();
  g->set_name("main");
  NodeProto *if_node = g->add_node();
  if_node->set_op_type("If");
  if_node->add_input("cond");
  // The single output is intentionally unnamed.
  if_node->add_output("");

  AttributeProto *then_attr = if_node->add_attribute();
  then_attr->set_name("then_branch");
  then_attr->set_type(AttributeProto::AttributeType::GRAPH);
  FillConstantBranch(*then_attr->add_g(), "then_g", "t", "z", 10.0f);

  AttributeProto *else_attr = if_node->add_attribute();
  else_attr->set_name("else_branch");
  else_attr->set_type(AttributeProto::AttributeType::GRAPH);
  FillConstantBranch(*else_attr->add_g(), "else_g", "e", "z", 1.0f);

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("cond", Tensor::FromBool("cond", {}, {1}));
  RunModelViaSession(model, rt);

  // The branch ran but its output name is empty, so nothing is propagated.
  EXPECT_FALSE(rt.Has(""));
  EXPECT_FALSE(rt.Has("z"));
}

// A sequence-state Loop whose tensor-state and scan outputs carry empty names
// must skip both when propagating results (RunLoopWithSequenceState).
TEST(RunLoopWithSequenceState, EmptyTensorStateAndScanOutputNamesSkipped) {
  RuntimeContext rt(KernelContext(DefaultOpset(13)));
  rt.Set("M", Tensor::FromInt64("M", {}, {3}));
  rt.Set("cond", Tensor::FromBool("cond", {}, {1}));
  rt.Set("acc_init", Tensor::FromFloat("acc_init", {}, {0.0f}));
  rt.PutSequence("seq_init", Sequence("seq_init", static_cast<int32_t>(DataType::FLOAT), {}));

  // Outputs are (acc_final, seq_final, scan); the tensor-state accumulator and
  // the scan output are left unnamed so both empty-name branches are exercised.
  RunNode(MakeLoopNode({"M", "cond", "acc_init", "seq_init"}, {"", "seq_final", ""},
                       BuildMixedSequenceLoopBody()),
          rt);

  ASSERT_TRUE(rt.HasSequence("seq_final"));
  const Sequence &seq = rt.GetSequence("seq_final");
  ASSERT_EQ(seq.size(), 3u);
  EXPECT_FALSE(rt.Has(""));
}

// A sequence-state Loop that runs zero iterations produces an empty scan output
// whose dtype/shape must be recovered from the body's declared output
// value-info (the zero-trip shape-patch branch of RunLoopWithSequenceState).
TEST(RunLoopWithSequenceState, ZeroTripScanOutputShapePatchedFromBodyValueInfo) {
  GraphProto body;
  body.set_name("zero_trip_body");
  body.add_input()->set_name("iter");
  body.add_input()->set_name("cond_in");
  AddSequenceFloatValueInfo(body.add_input(), "seq_in");

  {
    NodeProto *n = body.add_node();
    n->set_op_type("Identity");
    n->add_input("cond_in");
    n->add_output("cond_out");
  }
  AddIterAsFloat1D(body, "val");
  {
    NodeProto *n = body.add_node();
    n->set_op_type("SequenceInsert");
    n->add_input("seq_in");
    n->add_input("val");
    n->add_output("seq_out");
  }
  {
    NodeProto *n = body.add_node();
    n->set_op_type("Constant");
    n->add_output("scan_out");
    AttributeProto *a = n->add_attribute();
    a->set_name("value");
    a->set_type(AttributeProto::AttributeType::TENSOR);
    TensorProto *t = a->add_t();
    t->set_data_type(TensorProto::DataType::FLOAT);
    t->add_dims(2);
    t->add_float_data(0.0f);
    t->add_float_data(0.0f);
  }

  body.add_output()->set_name("cond_out");
  AddSequenceFloatValueInfo(body.add_output(), "seq_out");
  // Declare the per-iteration scan output type so the zero-trip path can patch
  // the stacked tensor's dtype (FLOAT) and trailing shape ([2]).
  ValueInfoProto *scan_vi = body.add_output();
  scan_vi->set_name("scan_out");
  TypeProto::Tensor *tt = scan_vi->add_type()->add_tensor_type();
  tt->set_elem_type(static_cast<int>(TensorProto::DataType::FLOAT));
  tt->add_shape()->add_dim()->set_dim_value(2);

  RuntimeContext rt(KernelContext(DefaultOpset(13)));
  // M = 0 forces zero trip iterations.
  rt.Set("M", Tensor::FromInt64("M", {}, {0}));
  rt.Set("cond", Tensor::FromBool("cond", {}, {1}));
  rt.PutSequence("seq_init", Sequence("seq_init", static_cast<int32_t>(DataType::FLOAT), {}));

  RunNode(MakeLoopNode({"M", "cond", "seq_init"}, {"seq_final", "scan"}, std::move(body)), rt);

  ASSERT_TRUE(rt.Has("scan"));
  const Tensor &scan = rt.Get("scan");
  EXPECT_EQ(scan.data_type, static_cast<int32_t>(DataType::FLOAT));
  EXPECT_EQ(scan.shape, (std::vector<int64_t>{0, 2}));
  EXPECT_EQ(scan.element_count(), 0);
}

// Scan opset 8 with a non-empty 'sequence_lens' input is unsupported and must
// raise an error.
TEST(RunModel, ScanOpset8NonEmptySequenceLensThrows) {
  ModelProto model;
  model.set_ir_version(3);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(8);

  GraphProto *g = model.add_graph();
  g->set_name("main");
  NodeProto *scan = g->add_node();
  scan->set_op_type("Scan");
  // Non-empty sequence_lens placeholder triggers the unsupported error.
  scan->add_input("seq_lens");
  scan->add_input("initial");
  scan->add_input("x");
  scan->add_output("y");
  scan->add_output("z");
  AttributeProto *num_attr = scan->add_attribute();
  num_attr->set_name("num_scan_inputs");
  num_attr->set_type(AttributeProto::AttributeType::INT);
  num_attr->set_i(1);

  AttributeProto *body_attr = scan->add_attribute();
  body_attr->set_name("body");
  body_attr->set_type(AttributeProto::AttributeType::GRAPH);
  GraphProto *body = body_attr->add_g();
  body->set_name("scan_body");
  body->add_input()->set_name("sum_in");
  body->add_input()->set_name("next");
  NodeProto *add = body->add_node();
  add->set_op_type("Add");
  add->add_input("sum_in");
  add->add_input("next");
  add->add_output("sum_out");
  NodeProto *id = body->add_node();
  id->set_op_type("Identity");
  id->add_input("sum_out");
  id->add_output("scan_out");
  body->add_output()->set_name("sum_out");
  body->add_output()->set_name("scan_out");

  RuntimeContext rt(KernelContext(DefaultOpset(8)));
  rt.Set("seq_lens", Tensor::FromInt32("seq_lens", {1}, {3}));
  rt.Set("initial", Tensor::FromFloat("initial", {1, 2}, {0.0f, 0.0f}));
  rt.Set("x", Tensor::FromFloat("x", {1, 3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));

  EXPECT_THROW(RunModelViaSession(model, rt), std::invalid_argument);
}

// Scan opset 8 requires all batched inputs to agree on the leading batch
// dimension; a mismatch must raise an error.
TEST(RunModel, ScanOpset8BatchDimMismatchThrows) {
  ModelProto model;
  model.set_ir_version(3);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(8);

  GraphProto *g = model.add_graph();
  g->set_name("main");
  NodeProto *scan = g->add_node();
  scan->set_op_type("Scan");
  scan->add_input("");
  scan->add_input("initial");
  scan->add_input("x");
  scan->add_output("y");
  scan->add_output("z");
  AttributeProto *num_attr = scan->add_attribute();
  num_attr->set_name("num_scan_inputs");
  num_attr->set_type(AttributeProto::AttributeType::INT);
  num_attr->set_i(1);

  AttributeProto *body_attr = scan->add_attribute();
  body_attr->set_name("body");
  body_attr->set_type(AttributeProto::AttributeType::GRAPH);
  GraphProto *body = body_attr->add_g();
  body->set_name("scan_body");
  body->add_input()->set_name("sum_in");
  body->add_input()->set_name("next");
  NodeProto *add = body->add_node();
  add->set_op_type("Add");
  add->add_input("sum_in");
  add->add_input("next");
  add->add_output("sum_out");
  NodeProto *id = body->add_node();
  id->set_op_type("Identity");
  id->add_input("sum_out");
  id->add_output("scan_out");
  body->add_output()->set_name("sum_out");
  body->add_output()->set_name("scan_out");

  RuntimeContext rt(KernelContext(DefaultOpset(8)));
  // State has batch dim 1 but the scan input has batch dim 2: disagreement.
  rt.Set("initial", Tensor::FromFloat("initial", {1, 2}, {0.0f, 0.0f}));
  rt.Set("x", Tensor::FromFloat(
                  "x", {2, 3, 2},
                  {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}));

  EXPECT_THROW(RunModelViaSession(model, rt), std::invalid_argument);
}

// A model-local function whose body node references an attribute that is
// neither supplied at the call-site nor declared as a default must have that
// referenced attribute dropped (BindRefAttributes erase path), leaving the op
// to fall back on its schema default.
TEST(RunModel, ModelLocalFunctionUnresolvedRefAttributeIsDropped) {
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(18);
  OperatorSetIdProto *custom_os = model.add_opset_import();
  custom_os->set_domain("custom");
  custom_os->set_version(1);

  FunctionProto *func = model.add_functions();
  func->set_name("Leaky");
  func->set_domain("custom");
  func->add_input("x");
  func->add_output("out");
  {
    NodeProto *n = func->add_node();
    n->set_op_type("LeakyRelu");
    n->add_input("x");
    n->add_output("out");
    // References a call-site attribute that is never provided; it is dropped so
    // LeakyRelu uses its default alpha (0.01).
    AttributeProto *alpha = n->add_attribute();
    alpha->set_name("alpha");
    alpha->set_ref_attr_name("missing_alpha");
    alpha->set_type(AttributeProto::AttributeType::FLOAT);
  }

  GraphProto *g = model.add_graph();
  g->set_name("test");
  NodeProto *call = g->add_node();
  call->set_op_type("Leaky");
  call->set_domain("custom");
  call->add_input("x");
  call->add_output("y");

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("x", Tensor::FromFloat("x", {1}, {-1.0f}));
  RunModelViaSession(model, rt);

  ASSERT_TRUE(rt.Has("y"));
  // Default alpha 0.01 applied to -1.0.
  EXPECT_NEAR(rt.Get("y").AsFloat()[0], -0.01f, 1e-6f);
}

// A model-local function body node with a plain (non-reference) sub-graph
// attribute containing a nested node that references a call-site attribute
// exercises the recursion into sub-graph attributes in BindRefAttributes.
TEST(RunModel, ModelLocalFunctionBindsRefAttributeInsideSubgraph) {
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(18);
  OperatorSetIdProto *custom_os = model.add_opset_import();
  custom_os->set_domain("custom");
  custom_os->set_version(1);

  FunctionProto *func = model.add_functions();
  func->set_name("CondLeaky");
  func->set_domain("custom");
  func->add_input("cond");
  func->add_input("x");
  func->add_output("out");
  func->add_attribute("alpha");
  {
    // If node with concrete (non-ref) branch graphs. The then-branch contains a
    // LeakyRelu node that references the call-site 'alpha' attribute, so
    // BindRefAttributes must recurse into the sub-graph to bind it.
    NodeProto *n = func->add_node();
    n->set_op_type("If");
    n->add_input("cond");
    n->add_output("out");

    AttributeProto *then_attr = n->add_attribute();
    then_attr->set_name("then_branch");
    then_attr->set_type(AttributeProto::AttributeType::GRAPH);
    GraphProto *then_g = then_attr->add_g();
    then_g->set_name("then_g");
    NodeProto *leaky = then_g->add_node();
    leaky->set_op_type("LeakyRelu");
    leaky->add_input("x");
    leaky->add_output("z");
    AttributeProto *alpha_ref = leaky->add_attribute();
    alpha_ref->set_name("alpha");
    alpha_ref->set_ref_attr_name("alpha");
    alpha_ref->set_type(AttributeProto::AttributeType::FLOAT);
    then_g->add_output()->set_name("z");

    AttributeProto *else_attr = n->add_attribute();
    else_attr->set_name("else_branch");
    else_attr->set_type(AttributeProto::AttributeType::GRAPH);
    GraphProto *else_g = else_attr->add_g();
    else_g->set_name("else_g");
    NodeProto *ident = else_g->add_node();
    ident->set_op_type("Identity");
    ident->add_input("x");
    ident->add_output("z");
    else_g->add_output()->set_name("z");
  }

  GraphProto *g = model.add_graph();
  g->set_name("test");
  NodeProto *call = g->add_node();
  call->set_op_type("CondLeaky");
  call->set_domain("custom");
  call->add_input("cond");
  call->add_input("x");
  call->add_output("y");
  AttributeProto *alpha = call->add_attribute();
  alpha->set_name("alpha");
  alpha->set_type(AttributeProto::AttributeType::FLOAT);
  alpha->set_f(0.5f);

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("cond", Tensor::FromBool("cond", {}, {1}));
  rt.Set("x", Tensor::FromFloat("x", {1}, {-2.0f}));
  RunModelViaSession(model, rt);

  ASSERT_TRUE(rt.Has("y"));
  // Bound alpha 0.5 applied to -2.0 via the then-branch LeakyRelu.
  EXPECT_NEAR(rt.Get("y").AsFloat()[0], -1.0f, 1e-6f);
}

// A model-local function call with an empty input name skips binding that
// formal parameter (the empty-name guard in ModelLocalFunctionKernel::Run).
// The parameter must be unused by the body for the call to succeed.
TEST(RunModel, ModelLocalFunctionEmptyInputNameSkipsBinding) {
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(18);
  OperatorSetIdProto *custom_os = model.add_opset_import();
  custom_os->set_domain("custom");
  custom_os->set_version(1);

  FunctionProto *func = model.add_functions();
  func->set_name("FirstOnly");
  func->set_domain("custom");
  func->add_input("a");
  func->add_input("b"); // unused
  func->add_output("out");
  {
    NodeProto *n = func->add_node();
    n->set_op_type("Identity");
    n->add_input("a");
    n->add_output("out");
  }

  GraphProto *g = model.add_graph();
  g->set_name("test");
  NodeProto *call = g->add_node();
  call->set_op_type("FirstOnly");
  call->set_domain("custom");
  call->add_input("x");
  call->add_input(""); // empty second input: binding skipped
  call->add_output("y");

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("x", Tensor::FromFloat("x", {2}, {3.0f, 4.0f}));
  RunModelViaSession(model, rt);

  ASSERT_TRUE(rt.Has("y"));
  const float *res = rt.Get("y").AsFloat();
  EXPECT_FLOAT_EQ(res[0], 3.0f);
  EXPECT_FLOAT_EQ(res[1], 4.0f);
}

// A model-local function call with an empty output name skips propagating that
// formal output back to the caller.
TEST(RunModel, ModelLocalFunctionEmptyOutputNameSkipsPropagation) {
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(18);
  OperatorSetIdProto *custom_os = model.add_opset_import();
  custom_os->set_domain("custom");
  custom_os->set_version(1);

  FunctionProto *func = model.add_functions();
  func->set_name("TwoOut");
  func->set_domain("custom");
  func->add_input("a");
  func->add_output("out0");
  func->add_output("out1");
  {
    NodeProto *n = func->add_node();
    n->set_op_type("Identity");
    n->add_input("a");
    n->add_output("out0");
  }
  {
    NodeProto *n = func->add_node();
    n->set_op_type("Identity");
    n->add_input("a");
    n->add_output("out1");
  }

  GraphProto *g = model.add_graph();
  g->set_name("test");
  NodeProto *call = g->add_node();
  call->set_op_type("TwoOut");
  call->set_domain("custom");
  call->add_input("x");
  call->add_output("y");
  call->add_output(""); // empty second output: propagation skipped

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("x", Tensor::FromFloat("x", {2}, {5.0f, 6.0f}));
  RunModelViaSession(model, rt);

  ASSERT_TRUE(rt.Has("y"));
  EXPECT_FLOAT_EQ(rt.Get("y").AsFloat()[0], 5.0f);
  EXPECT_FALSE(rt.Has(""));
}

// A model-local function defined in the default (empty) domain is registered
// under FunctionLookupKey's empty-domain branch (":<name>:<overload>").
TEST(RunModel, DefaultDomainModelLocalFunctionLookupKey) {
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(18);

  FunctionProto *func = model.add_functions();
  func->set_name("MyDefaultDomainFn");
  func->set_domain(""); // default (empty) domain
  func->add_input("a");
  func->add_output("out");
  {
    NodeProto *n = func->add_node();
    n->set_op_type("Identity");
    n->add_input("a");
    n->add_output("out");
  }

  GraphProto *g = model.add_graph();
  g->set_name("test");
  NodeProto *node = g->add_node();
  node->set_op_type("Identity");
  node->add_input("x");
  node->add_output("y");

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  RegisterModelFunctions(model, rt);

  // FunctionLookupKey formats an empty domain as a leading ':' separator.
  ASSERT_EQ(rt.functions().count(":MyDefaultDomainFn:"), 1u);
  EXPECT_EQ(rt.functions().at(":MyDefaultDomainFn:"), func);
}

// A model-local function body node carrying a repeated GRAPHS attribute whose
// nested graphs contain reference attributes exercises the ``graphs()``
// recursion branch of BindRefAttributes. The host node (Identity) ignores the
// extra attribute at run time.
TEST(RunModel, ModelLocalFunctionBindsRefAttributeInsideGraphsAttribute) {
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(18);
  OperatorSetIdProto *custom_os = model.add_opset_import();
  custom_os->set_domain("custom");
  custom_os->set_version(1);

  FunctionProto *func = model.add_functions();
  func->set_name("GraphsAttr");
  func->set_domain("custom");
  func->add_input("a");
  func->add_output("out");
  func->add_attribute("alpha");
  {
    NodeProto *n = func->add_node();
    n->set_op_type("Identity");
    n->add_input("a");
    n->add_output("out");
    // A repeated-graphs attribute whose single graph contains a node that
    // references the call-site 'alpha' attribute. BindRefAttributes recurses
    // into every graph of the attribute to bind such references.
    AttributeProto *graphs_attr = n->add_attribute();
    graphs_attr->set_name("extra_graphs");
    graphs_attr->set_type(AttributeProto::AttributeType::GRAPHS);
    GraphProto *sub = graphs_attr->add_graphs();
    sub->set_name("sub_g");
    NodeProto *inner = sub->add_node();
    inner->set_op_type("LeakyRelu");
    inner->add_input("a");
    inner->add_output("z");
    AttributeProto *alpha_ref = inner->add_attribute();
    alpha_ref->set_name("alpha");
    alpha_ref->set_ref_attr_name("alpha");
    alpha_ref->set_type(AttributeProto::AttributeType::FLOAT);
    sub->add_output()->set_name("z");
  }

  GraphProto *g = model.add_graph();
  g->set_name("test");
  NodeProto *call = g->add_node();
  call->set_op_type("GraphsAttr");
  call->set_domain("custom");
  call->add_input("x");
  call->add_output("y");
  AttributeProto *alpha = call->add_attribute();
  alpha->set_name("alpha");
  alpha->set_type(AttributeProto::AttributeType::FLOAT);
  alpha->set_f(0.25f);

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.Set("x", Tensor::FromFloat("x", {2}, {1.0f, 2.0f}));
  RunModelViaSession(model, rt);

  ASSERT_TRUE(rt.Has("y"));
  // Identity passes the input through unchanged.
  EXPECT_FLOAT_EQ(rt.Get("y").AsFloat()[0], 1.0f);
  EXPECT_FLOAT_EQ(rt.Get("y").AsFloat()[1], 2.0f);
}

// A SequenceMap over a STRING sequence forces the STRING branch of
// CloneTensor when the body's per-iteration outputs are cloned.
TEST(RunModel, SequenceMapOverStringSequenceClonesStrings) {
  GraphProto body;
  body.set_name("seq_map_body");
  body.add_input()->set_name("x_in");
  {
    NodeProto *n = body.add_node();
    n->set_op_type("Identity");
    n->add_input("x_in");
    n->add_output("y_out");
  }
  body.add_output()->set_name("y_out");

  NodeProto seq_map_node = MakeNode("SequenceMap", {"xs"}, {"ys"});
  AttributeProto *body_attr = seq_map_node.add_attribute();
  body_attr->set_name("body");
  body_attr->set_type(AttributeProto::AttributeType::GRAPH);
  *body_attr->add_g() = body;

  GraphProto graph;
  graph.set_name("main");
  ValueInfoProto vi_xs, vi_ys;
  vi_xs.set_name("xs");
  vi_ys.set_name("ys");
  graph.ref_input().push_back(vi_xs);
  graph.ref_output().push_back(vi_ys);
  graph.ref_node().push_back(seq_map_node);

  RuntimeContext rt(KernelContext(DefaultOpset(18)));
  rt.PutSequence("xs", Sequence("xs", static_cast<int32_t>(DataType::STRING),
                                {Tensor::FromStrings("x0", {1}, {"alpha"}),
                                 Tensor::FromStrings("x1", {1}, {"beta"})}));

  RunGraphViaSession(graph, rt);

  ASSERT_TRUE(rt.HasSequence("ys"));
  const Sequence &ys = rt.GetSequence("ys");
  ASSERT_EQ(ys.size(), 2u);
  EXPECT_EQ(ys.at(0).AsStrings()[0], "alpha");
  EXPECT_EQ(ys.at(1).AsStrings()[0], "beta");
}

// Verbose progress logging while running a sub-graph body node must include the
// "<attr_name>@<node_index>/" prefix (the subgraph-index branch of
// PrintNodeProgress).
TEST(RunModel, VerboseProgressInsideSubgraphLogsAttrIndexPrefix) {
  const uint64_t time_seed =
      static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
  const uint64_t thread_seed =
      static_cast<uint64_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
  std::mt19937_64 rng(time_seed ^ thread_seed);
  const std::filesystem::path log_path =
      std::filesystem::temp_directory_path() /
      ("onnx_light_verbose_subgraph_" + std::to_string(rng()) + ".log");
  TempFileCleanupGuard cleanup(log_path);
  EnvVarGuard guard("ONNX_LIGHT_LOG");
#ifdef _WIN32
  _putenv_s("ONNX_LIGHT_LOG", log_path.string().c_str());
#else
  setenv("ONNX_LIGHT_LOG", log_path.string().c_str(), 1);
#endif

  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_version(18);

  GraphProto *g = model.add_graph();
  g->set_name("main");
  NodeProto *if_node = g->add_node();
  if_node->set_op_type("If");
  if_node->add_input("cond");
  if_node->add_output("out");

  AttributeProto *then_attr = if_node->add_attribute();
  then_attr->set_name("then_branch");
  then_attr->set_type(AttributeProto::AttributeType::GRAPH);
  FillConstantBranch(*then_attr->add_g(), "then_g", "t", "z", 10.0f);

  AttributeProto *else_attr = if_node->add_attribute();
  else_attr->set_name("else_branch");
  else_attr->set_type(AttributeProto::AttributeType::GRAPH);
  FillConstantBranch(*else_attr->add_g(), "else_g", "e", "z", 1.0f);

  RuntimeContext rt(KernelContext(DefaultOpset(18)),
                    core::runtime::RuntimeContextOptions{.verbose = 1});
  rt.Set("cond", Tensor::FromBool("cond", {}, {1}));
  RunModelViaSession(model, rt);

  ASSERT_TRUE(std::filesystem::exists(log_path));
  std::ifstream in(log_path);
  ASSERT_TRUE(in.is_open());
  std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  // The Add node inside the taken then-branch is logged with the subgraph
  // prefix "then_branch@<index>/".
  EXPECT_NE(content.find("then_branch@"), std::string::npos);
}

// The convenience ``RunModel`` helper runs a whole model given named input
// tensors and returns its declared outputs, seeding both external inputs and
// the graph's initializers automatically.
TEST(RunModel, RunModelTensorInputsReturnsOutputs) {
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_domain("");
  os->set_version(18);

  GraphProto *graph = model.add_graph();
  graph->set_name("main");
  graph->add_input()->set_name("x");
  graph->add_input()->set_name("z");
  graph->add_output()->set_name("y");

  // y = Abs(x) + z, mirroring the shared model used by the Python tests.
  NodeProto *abs_node = graph->add_node();
  abs_node->set_op_type("Abs");
  abs_node->add_input("x");
  abs_node->add_output("t");
  NodeProto *add_node = graph->add_node();
  add_node->set_op_type("Add");
  add_node->add_input("t");
  add_node->add_input("z");
  add_node->add_output("y");

  Tensors inputs;
  inputs.push_back(Tensor::FromFloat("x", {3}, {1.0f, -2.0f, 3.0f}));
  inputs.push_back(Tensor::FromFloat("z", {3}, {10.0f, 20.0f, 30.0f}));

  Tensors outputs = RunModel(model, std::move(inputs));
  ASSERT_EQ(outputs.size(), 1u);
  EXPECT_EQ(outputs[0].name, "y");
  EXPECT_FALSE(outputs[0].is_borrowed());
  ASSERT_EQ(outputs[0].element_count(), 3);
  const float *got = outputs[0].AsFloat();
  EXPECT_FLOAT_EQ(got[0], 11.0f);
  EXPECT_FLOAT_EQ(got[1], 22.0f);
  EXPECT_FLOAT_EQ(got[2], 33.0f);
}

// ``RunModel`` seeds the graph's initializers so a model needs only its
// non-initializer inputs supplied.
TEST(RunModel, RunModelSeedsInitializers) {
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *os = model.add_opset_import();
  os->set_domain("");
  os->set_version(18);

  GraphProto *graph = model.add_graph();
  graph->set_name("main");
  graph->add_input()->set_name("x");
  graph->add_output()->set_name("y");

  // ``w`` is an initializer, only ``x`` is supplied at run time.
  const float weights[2] = {2.0f, 3.0f};
  TensorProto *init = graph->add_initializer();
  init->set_name("w");
  init->set_data_type(TensorProto::DataType::FLOAT);
  init->add_dims(2);
  std::vector<uint8_t> raw(sizeof(weights));
  std::memcpy(raw.data(), weights, sizeof(weights));
  init->set_raw_data(utils::ByteSpan(raw));

  NodeProto *node = graph->add_node();
  node->set_op_type("Add");
  node->add_input("x");
  node->add_input("w");
  node->add_output("y");

  Tensors inputs;
  inputs.push_back(Tensor::FromFloat("x", {2}, {1.0f, 1.0f}));
  Tensors outputs = RunModel(model, std::move(inputs));
  ASSERT_EQ(outputs.size(), 1u);
  ASSERT_EQ(outputs[0].element_count(), 2);
  const float *got = outputs[0].AsFloat();
  EXPECT_FLOAT_EQ(got[0], 3.0f);
  EXPECT_FLOAT_EQ(got[1], 4.0f);
}

// ``RunModel`` rejects a model without a graph.
TEST(RunModel, RunModelWithoutGraphThrows) {
  ModelProto model;
  model.set_ir_version(10);
  EXPECT_THROW(RunModel(model, Tensors{}), std::invalid_argument);
}

} // namespace Test