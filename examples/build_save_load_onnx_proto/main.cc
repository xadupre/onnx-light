/**
 * main.cc -- Standalone example: build, save and load an ONNX model
 * using only the onnx_light proto layer (lib_onnx_proto). No schemas,
 * no shape inference, no checker, no backend kernels are required.
 *
 * Usage:
 *   ./build_save_load_onnx_proto [output_path]
 *
 * The executable builds a tiny single-node ``Y = Add(X, B)`` model with a
 * FLOAT[3] initializer, serializes it with FileWriteStream +
 * SerializeModelProtoToStream, parses it back with FileStream +
 * ParseModelProtoFromStream, and verifies that the round-trip matches.
 *
 * See CMakeLists.txt for build instructions.
 */

#include "onnx.h"
#include "onnx_helper.h"
#include "stream.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace onnx_light = ONNX_LIGHT_NAMESPACE;

// Builds the model ``Y = Add(X, B)`` where ``B`` is a FLOAT[3] initializer.
onnx_light::ModelProto BuildAddModel(const std::vector<float> &b_values) {
  onnx_light::ModelProto model;
  model.set_ir_version(7);
  model.set_producer_name("build_save_load_onnx_proto");

  onnx_light::OperatorSetIdProto *opset = model.add_opset_import();
  opset->set_domain("");
  opset->set_version(17);

  onnx_light::GraphProto &graph = model.ref_graph();
  graph.set_name("add_graph");

  // Graph input X: FLOAT[3]
  onnx_light::ValueInfoProto *x = graph.add_input();
  x->set_name("X");
  onnx_light::TypeProto::Tensor &x_type = x->ref_type().ref_tensor_type();
  x_type.set_elem_type(onnx_light::TensorProto::DataType::FLOAT);
  onnx_light::TensorShapeProto::Dimension *x_dim = x_type.ref_shape().add_dim();
  x_dim->set_dim_value(static_cast<int64_t>(b_values.size()));

  // Graph output Y: FLOAT[3]
  onnx_light::ValueInfoProto *y = graph.add_output();
  y->set_name("Y");
  onnx_light::TypeProto::Tensor &y_type = y->ref_type().ref_tensor_type();
  y_type.set_elem_type(onnx_light::TensorProto::DataType::FLOAT);
  onnx_light::TensorShapeProto::Dimension *y_dim = y_type.ref_shape().add_dim();
  y_dim->set_dim_value(static_cast<int64_t>(b_values.size()));

  // Initializer B: FLOAT[3]
  onnx_light::TensorProto *b = graph.add_initializer();
  b->set_name("B");
  b->set_data_type(onnx_light::TensorProto::DataType::FLOAT);
  b->add_dims(static_cast<uint64_t>(b_values.size()));
  for (float v : b_values) {
    b->add_float_data(v);
  }

  // Node: Y = Add(X, B)
  onnx_light::NodeProto node = onnx_light::MakeNode("Add", {"X", "B"}, {"Y"}, nullptr, "add_node");
  graph.add_node(std::move(node));

  return model;
}

void PrintModelSummary(const onnx_light::ModelProto &model) {
  if (model.has_ir_version()) {
    std::cout << "  IR version       : " << model.ref_ir_version() << "\n";
  }
  if (model.has_producer_name()) {
    std::cout << "  Producer name    : " << model.ref_producer_name().as_string() << "\n";
  }
  if (model.has_graph()) {
    const onnx_light::GraphProto &graph = model.ref_graph();
    std::cout << "  Graph name       : " << graph.ref_name().as_string() << "\n";
    std::cout << "  Nodes            : " << graph.ref_node().size() << "\n";
    std::cout << "  Inputs           : " << graph.ref_input().size() << "\n";
    std::cout << "  Outputs          : " << graph.ref_output().size() << "\n";
    std::cout << "  Initializers     : " << graph.ref_initializer().size() << "\n";
  }
}

// Reads the FLOAT[3] values of initializer ``B`` from the loaded model.
std::vector<float> ExtractInitializerB(const onnx_light::ModelProto &model) {
  if (!model.has_graph()) {
    throw std::runtime_error("Loaded model has no graph.");
  }
  const onnx_light::GraphProto &graph = model.ref_graph();
  for (size_t i = 0; i < graph.ref_initializer().size(); ++i) {
    const onnx_light::TensorProto &t = graph.ref_initializer()[i];
    if (t.ref_name().as_string() != "B") {
      continue;
    }
    if (t.ref_data_type() != onnx_light::TensorProto::DataType::FLOAT) {
      throw std::runtime_error("Initializer 'B' is not FLOAT.");
    }
    // The values can be stored in ``float_data`` (typed field, used when
    // we serialize the model we just built) or in ``raw_data`` (a contiguous
    // little-endian byte buffer).
    if (t.ref_float_data().size() > 0) {
      std::vector<float> out;
      out.reserve(t.ref_float_data().size());
      for (size_t k = 0; k < t.ref_float_data().size(); ++k) {
        out.push_back(t.ref_float_data()[k]);
      }
      return out;
    }
    if (t.has_raw_data()) {
      const onnx_light::utils::ByteSpan &raw = t.ref_raw_data();
      const size_t count = raw.size() / sizeof(float);
      std::vector<float> out(count);
      std::memcpy(out.data(), raw.data(), count * sizeof(float));
      return out;
    }
    throw std::runtime_error("Initializer 'B' has no float_data nor raw_data.");
  }
  throw std::runtime_error("Initializer 'B' not found in loaded model.");
}

} // namespace

int main(int argc, char *argv[]) {
  if (argc > 2) {
    std::cerr << "Usage: " << argv[0] << " [output_path]\n";
    return 1;
  }

  const std::string output_path = (argc == 2) ? argv[1] : "example_add.onnx";
  const std::vector<float> b_values = {1.0f, 2.0f, 3.0f};

  // 1. Build the model in memory.
  onnx_light::ModelProto model = BuildAddModel(b_values);

  // 2. Serialize it to disk.
  try {
    onnx_light::utils::FileWriteStream stream(output_path);
    onnx_light::SerializeOptions opts;
    onnx_light::SerializeModelProtoToStream(model, stream, opts);
  } catch (const std::exception &e) {
    std::cerr << "Error saving '" << output_path << "': " << e.what() << "\n";
    return 1;
  }
  std::cout << "Saved: " << output_path << "\n";
  PrintModelSummary(model);

  // 3. Parse it back from disk.
  onnx_light::ModelProto loaded;
  try {
    onnx_light::utils::FileStream stream(output_path);
    onnx_light::ParseOptions parse_opts;
    onnx_light::ParseModelProtoFromStream(loaded, stream, parse_opts);
  } catch (const std::exception &e) {
    std::cerr << "Error loading '" << output_path << "': " << e.what() << "\n";
    return 1;
  }
  std::cout << "Loaded: " << output_path << "\n";
  PrintModelSummary(loaded);

  // 4. Verify the round-trip on the initializer values.
  const std::vector<float> loaded_b = ExtractInitializerB(loaded);
  if (loaded_b.size() != b_values.size()) {
    std::cerr << "Initializer 'B' size mismatch: expected " << b_values.size() << ", got "
              << loaded_b.size() << "\n";
    return 1;
  }
  std::cout << "  Initializer B    : [";
  for (size_t i = 0; i < loaded_b.size(); ++i) {
    if (i != 0) {
      std::cout << ", ";
    }
    std::cout << loaded_b[i];
  }
  std::cout << "]\n";
  for (size_t i = 0; i < loaded_b.size(); ++i) {
    if (std::fabs(loaded_b[i] - b_values[i]) > 0.0f) {
      std::cerr << "Initializer 'B' value mismatch at index " << i << ": expected " << b_values[i]
                << ", got " << loaded_b[i] << "\n";
      return 1;
    }
  }

  std::cout << "Round-trip OK\n";
  return 0;
}
