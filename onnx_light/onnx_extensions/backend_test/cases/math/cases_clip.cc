// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/random.h"
#include "onnx_extensions/backend_test/cases/math/include_math_cases.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include <cstdint>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

namespace {

// Builds a Clip node with the supplied input names. Pass an empty string in
// any of ``min_name``/``max_name`` to declare the optional input as absent
// (matching upstream ONNX, e.g. ``onnx.helper.make_node("Clip", ["x", "",
// "max"], ...)``).
NodeProto MakeClipNode(const std::string &min_name = "min", const std::string &max_name = "max") {
  NodeProto node;
  node.set_op_type("Clip");
  node.add_input("x");
  node.add_input(min_name);
  node.add_input(max_name);
  node.add_output("y");
  return node;
}

// Variant of ``MakeClipNode`` for the ``x``-only case (no min/max inputs).
NodeProto MakeClipNodeNoBounds() {
  NodeProto node;
  node.set_op_type("Clip");
  node.add_input("x");
  node.add_output("y");
  return node;
}

template <typename T> Tensor ScalarTensor(T value);

template <> Tensor ScalarTensor<float>(float value) { return Tensor::FromFloat("", {}, {value}); }
template <> Tensor ScalarTensor<int8_t>(int8_t value) { return Tensor::FromInt8("", {}, {value}); }

// Casts a vector of floats element-wise to int8_t. Used to mirror the
// upstream ``np.random.randn(...).astype(np.int8)`` pattern from
// ``onnx.backend.test.case.node.clip``.
std::vector<int8_t> ToInt8(const std::vector<float> &v) {
  std::vector<int8_t> out;
  out.reserve(v.size());
  for (float f : v) {
    out.push_back(static_cast<int8_t>(f));
  }
  return out;
}

} // namespace

// ---------------------------------------------------------------------------
// Clip — y = min(max(x, min), max) (since opset 6, min/max as optional
// inputs since opset 11, widened to all numeric types in opset 12). Mirrors
// the upstream ``onnx.backend.test.case.node.clip.Clip`` cases.
// ---------------------------------------------------------------------------
void RegisterClipCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(13);
  const KernelContext ctx{opset};
  const onnx_kernels::kernel::Clip clip_kernel{ctx};

  if (mode == TestMode::BENCHMARK) {
    NodeProto node = MakeClipNode();
    const int64_t size = kBenchmarkElementwiseSize;
    Expect(registry, std::move(node), "test_cc_clip_benchmark", {opset}, {size, 1, 1}, {size},
           [clip_kernel, size]() -> IoData {
             Tensor x = RandnTensor(DataType::FLOAT, {size}, /*seed=*/2001);
             Tensor min_val = ScalarTensor<float>(-1.0f);
             Tensor max_val = ScalarTensor<float>(1.0f);
             Tensor y = clip_kernel(x, &min_val, &max_val);
             return IoData{{std::move(x), std::move(min_val), std::move(max_val)}, {std::move(y)}};
           });
    return;
  }

  // Deterministic onnx-light-specific case (``test_cc_*``).
  {
    NodeProto node = MakeClipNode();
    Expect(registry, std::move(node), "test_cc_clip", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {5}, {-2.0f, -0.5f, 0.0f, 0.5f, 2.0f});
      Tensor min_val = ScalarTensor<float>(-1.0f);
      Tensor max_val = ScalarTensor<float>(1.0f);
      Tensor y = clip_kernel(x, &min_val, &max_val);
      return IoData{{std::move(x), std::move(min_val), std::move(max_val)}, {std::move(y)}};
    });
  }

  // Upstream ONNX ``Clip.export()`` cases — float min/max in [-1, 1].
  {
    NodeProto node = MakeClipNode();
    Expect(registry, std::move(node), "test_clip_example", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {3}, {-2.0f, 0.0f, 2.0f});
      Tensor min_val = ScalarTensor<float>(-1.0f);
      Tensor max_val = ScalarTensor<float>(1.0f);
      Tensor y = clip_kernel(x, &min_val, &max_val);
      return IoData{{std::move(x), std::move(min_val), std::move(max_val)}, {std::move(y)}};
    });
  }
  {
    NodeProto node = MakeClipNode();
    Expect(registry, std::move(node), "test_clip", {opset}, [=]() -> IoData {
      const std::vector<int64_t> shape = {3, 4, 5};
      Tensor x = RandnTensor(DataType::FLOAT, shape, /*seed=*/1);
      Tensor min_val = ScalarTensor<float>(-1.0f);
      Tensor max_val = ScalarTensor<float>(1.0f);
      Tensor y = clip_kernel(x, &min_val, &max_val);
      return IoData{{std::move(x), std::move(min_val), std::move(max_val)}, {std::move(y)}};
    });
  }

  // float min/max in [-5, 5] — inbounds / outbounds / splitbounds.
  {
    NodeProto node = MakeClipNode();
    Expect(registry, std::move(node), "test_clip_inbounds", {opset}, [=]() -> IoData {
      Tensor min_val = ScalarTensor<float>(-5.0f);
      Tensor max_val = ScalarTensor<float>(5.0f);
      Tensor x = Tensor::FromFloat("", {3}, {-1.0f, 0.0f, 1.0f});
      Tensor y = clip_kernel(x, &min_val, &max_val);
      return IoData{{std::move(x), std::move(min_val), std::move(max_val)}, {std::move(y)}};
    });
  }
  {
    NodeProto node = MakeClipNode();
    Expect(registry, std::move(node), "test_clip_outbounds", {opset}, [=]() -> IoData {
      Tensor min_val = ScalarTensor<float>(-5.0f);
      Tensor max_val = ScalarTensor<float>(5.0f);
      Tensor x = Tensor::FromFloat("", {3}, {-6.0f, 0.0f, 6.0f});
      Tensor y = clip_kernel(x, &min_val, &max_val);
      return IoData{{std::move(x), std::move(min_val), std::move(max_val)}, {std::move(y)}};
    });
  }
  {
    NodeProto node = MakeClipNode();
    Expect(registry, std::move(node), "test_clip_splitbounds", {opset}, [=]() -> IoData {
      Tensor min_val = ScalarTensor<float>(-5.0f);
      Tensor max_val = ScalarTensor<float>(5.0f);
      Tensor x = Tensor::FromFloat("", {3}, {-1.0f, 0.0f, 6.0f});
      Tensor y = clip_kernel(x, &min_val, &max_val);
      return IoData{{std::move(x), std::move(min_val), std::move(max_val)}, {std::move(y)}};
    });
  }

  // min > max: per the ONNX spec, all input values are replaced by max.
  {
    NodeProto node = MakeClipNode();
    Expect(registry, std::move(node), "test_clip_min_greater_than_max", {opset}, [=]() -> IoData {
      Tensor min_val = ScalarTensor<float>(2.0f);
      Tensor max_val = ScalarTensor<float>(1.0f);
      Tensor x = Tensor::FromFloat("", {3}, {-2.0f, 0.0f, 6.0f});
      Tensor y = clip_kernel(x, &min_val, &max_val);
      return IoData{{std::move(x), std::move(min_val), std::move(max_val)}, {std::move(y)}};
    });
  }

  // float default min / max / inbounds — exercise optional inputs.
  {
    // Inputs: ``["x", "min"]`` — trailing optional ``max`` omitted entirely.
    NodeProto node;
    node.set_op_type("Clip");
    node.add_input("x");
    node.add_input("min");
    node.add_output("y");
    Expect(registry, std::move(node), "test_clip_default_min", {opset}, [=]() -> IoData {
      Tensor min_val = ScalarTensor<float>(0.0f);
      const std::vector<int64_t> shape = {3, 4, 5};
      Tensor x = RandnTensor(DataType::FLOAT, shape, /*seed=*/2);
      Tensor y = clip_kernel(x, &min_val, /*max=*/nullptr);
      return IoData{{std::move(x), std::move(min_val)}, {std::move(y)}};
    });
  }
  {
    // Inputs: ``["x", "", "max"]`` — missing min, present max.
    NodeProto node;
    node.set_op_type("Clip");
    node.add_input("x");
    node.add_input("");
    node.add_input("max");
    node.add_output("y");
    Expect(registry, std::move(node), "test_clip_default_max", {opset}, [=]() -> IoData {
      Tensor max_val = ScalarTensor<float>(0.0f);
      const std::vector<int64_t> shape = {3, 4, 5};
      Tensor x = RandnTensor(DataType::FLOAT, shape, /*seed=*/3);
      Tensor y = clip_kernel(x, /*min=*/nullptr, &max_val);
      return IoData{{std::move(x), std::move(max_val)}, {std::move(y)}};
    });
  }
  {
    NodeProto node = MakeClipNodeNoBounds();
    Expect(registry, std::move(node), "test_clip_default_inbounds", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {3}, {-1.0f, 0.0f, 1.0f});
      Tensor y = clip_kernel(x, /*min=*/nullptr, /*max=*/nullptr);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // int8 variants: opset 12 widened ``T`` to all numeric tensors.
  const OpsetId opset12 = DefaultOpset(12);
  const KernelContext ctx12{opset12};
  const onnx_kernels::kernel::Clip clip_kernel12{ctx12};
  {
    // Inputs: ``["x", "min"]`` — trailing optional ``max`` omitted entirely.
    NodeProto node;
    node.set_op_type("Clip");
    node.add_input("x");
    node.add_input("min");
    node.add_output("y");
    Expect(registry, std::move(node), "test_clip_default_int8_min", {opset12}, [=]() -> IoData {
      Tensor min_val = ScalarTensor<int8_t>(0);
      const std::vector<int64_t> shape = {3, 4, 5};
      Tensor x = Tensor::FromInt8("", shape, ToInt8(Randn<float>(shape, /*seed=*/4)));
      Tensor y = clip_kernel12(x, &min_val, /*max=*/nullptr);
      return IoData{{std::move(x), std::move(min_val)}, {std::move(y)}};
    });
  }
  {
    NodeProto node;
    node.set_op_type("Clip");
    node.add_input("x");
    node.add_input("");
    node.add_input("max");
    node.add_output("y");
    Expect(registry, std::move(node), "test_clip_default_int8_max", {opset12}, [=]() -> IoData {
      Tensor max_val = ScalarTensor<int8_t>(0);
      const std::vector<int64_t> shape = {3, 4, 5};
      Tensor x = Tensor::FromInt8("", shape, ToInt8(Randn<float>(shape, /*seed=*/5)));
      Tensor y = clip_kernel12(x, /*min=*/nullptr, &max_val);
      return IoData{{std::move(x), std::move(max_val)}, {std::move(y)}};
    });
  }
  {
    NodeProto node = MakeClipNodeNoBounds();
    Expect(registry, std::move(node), "test_clip_default_int8_inbounds", {opset12},
           [=]() -> IoData {
             Tensor x = Tensor::FromInt8("", {3}, {-1, 0, 1});
             Tensor y = clip_kernel12(x, /*min=*/nullptr, /*max=*/nullptr);
             return IoData{{std::move(x)}, {std::move(y)}};
           });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
