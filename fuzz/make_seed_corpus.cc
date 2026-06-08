// Copyright (c) ONNX Project Contributors
// SPDX-License-Identifier: Apache-2.0
//
// Seed corpus generator for the libFuzzer harnesses living next to
// this file. Replaces the former
// ``onnx_light/fuzz/make_seed_corpus.py``: writes one seed file per
// model into the requested output directories so that they can be
// passed directly to libFuzzer (``./fuzz_target <corpus_dir>``) or
// packaged by OSS-Fuzz's build scripts.
//
// Usage:
//   make_seed_corpus <version_converter_out_dir>
//                    <parser_out_dir>
//                    <shape_inference_out_dir>
//
// All seeds are authored in ONNX text format and serialized to bytes
// via ``OnnxParser``, so the seed bodies stay close to what a human
// would read in the upstream OSS-Fuzz seed-corpus zips.

#include "onnx_lib/defs/parser.h"
#include "onnx_proto/onnx.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

using ONNX_LIGHT_NAMESPACE::ModelProto;
using ONNX_LIGHT_NAMESPACE::OnnxParser;

namespace {

std::string text_to_serialized_model(const char *text) {
  ModelProto m;
  auto status = OnnxParser::Parse(m, text);
  if (!status.IsOK()) {
    throw std::runtime_error(std::string("seed parse failed: ") + status.ErrorMessage());
  }
  std::string out;
  m.SerializeToString(out);
  return out;
}

void write_file(const std::filesystem::path &dir, const std::string &name,
                const std::string &bytes) {
  std::filesystem::create_directories(dir);
  std::ofstream f(dir / name, std::ios::binary);
  if (!f) {
    throw std::runtime_error("failed to open " + (dir / name).string());
  }
  f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

} // namespace

int main(int argc, char *argv[]) {
  if (argc != 4) {
    std::cerr << "Usage: " << (argc > 0 ? argv[0] : "make_seed_corpus")
              << " <version_converter_out_dir> <parser_out_dir>"
                 " <shape_inference_out_dir>\n";
    return 2;
  }
  const std::filesystem::path vc_dir = argv[1];
  const std::filesystem::path parser_dir = argv[2];
  const std::filesystem::path si_dir = argv[3];

  try {
    // version_converter seeds: a handful of single-op models that
    // exercise the per-opset converter dispatch.
    write_file(vc_dir, "softmax_12.onnx",
               text_to_serialized_model("<\n"
                                        "  ir_version: 7,\n"
                                        "  opset_import: [\"\" : 12]\n"
                                        ">\n"
                                        "agraph (float[N, 4] X) => (float[N, 4] Y)\n"
                                        "{ Y = Softmax(X) }\n"));
    write_file(vc_dir, "softmax_13.onnx",
               text_to_serialized_model("<\n"
                                        "  ir_version: 7,\n"
                                        "  opset_import: [\"\" : 13]\n"
                                        ">\n"
                                        "agraph (float[N, 4] X) => (float[N, 4] Y)\n"
                                        "{ Y = Softmax(X) }\n"));
    write_file(vc_dir, "cast_9_to_float.onnx",
               text_to_serialized_model("<\n"
                                        "  ir_version: 7,\n"
                                        "  opset_import: [\"\" : 9]\n"
                                        ">\n"
                                        "agraph (int64[N] X) => (float[N] Y)\n"
                                        "{ Y = Cast<to=1>(X) }\n"));
    write_file(vc_dir, "relu_15.onnx",
               text_to_serialized_model("<\n"
                                        "  ir_version: 7,\n"
                                        "  opset_import: [\"\" : 15]\n"
                                        ">\n"
                                        "agraph (float[N] X) => (float[N] Y)\n"
                                        "{ Y = Relu(X) }\n"));
    write_file(vc_dir, "sigmoid_15.onnx",
               text_to_serialized_model("<\n"
                                        "  ir_version: 7,\n"
                                        "  opset_import: [\"\" : 15]\n"
                                        ">\n"
                                        "agraph (float[N] X) => (float[N] Y)\n"
                                        "{ Y = Sigmoid(X) }\n"));

    // parser seeds: textual ONNX models exercising the productions
    // used by ``OnnxParser::Parse<ModelProto>`` directly. Written as
    // raw text — the parser harness consumes them with the same UTF-8
    // interpretation it applies to fuzzer-mutated inputs.
    write_file(parser_dir, "basic_matmul_softmax.txt",
               "<\n"
               "  ir_version: 7,\n"
               "  opset_import: [\"\" : 10]\n"
               ">\n"
               "agraph (float[N, 128] X, float[128, 10] W, float[10] B) => (float[N] C)\n"
               "{\n"
               "   T = MatMul(X, W)\n"
               "   S = Add(T, B)\n"
               "   C = Softmax(S)\n"
               "}\n");
    write_file(parser_dir, "multi_opset.txt",
               "<\n"
               "  ir_version: 7,\n"
               "  opset_import: [\"\" : 10, \"com.microsoft\" : 1]\n"
               ">\n"
               "agraph (float[N] X) => (float[N] Y)\n"
               "{ Y = Relu(X) }\n");
    write_file(parser_dir, "model_with_metadata.txt",
               "<\n"
               "  ir_version: 9,\n"
               "  opset_import: [\"\" : 15],\n"
               "  producer_name: \"oss-fuzz-seed\",\n"
               "  producer_version: \"1.0\",\n"
               "  model_version: 1,\n"
               "  doc_string: \"seed model for fuzz_parser\"\n"
               ">\n"
               "agraph (float[N] x) => (float[N] y)\n"
               "{ y = Relu(x) }\n");

    // shape-inference seeds: serialized ModelProtos. The harness
    // loads each seed via ParseFromString and then runs InferShapes.
    write_file(si_dir, "linear_relu_sigmoid.onnx",
               text_to_serialized_model("<\n"
                                        "  ir_version: 7,\n"
                                        "  opset_import: [\"\" : 15]\n"
                                        ">\n"
                                        "agraph (float[1, 4] X) => (float[1, 4] Y)\n"
                                        "{\n"
                                        "   T = Relu(X)\n"
                                        "   Y = Sigmoid(T)\n"
                                        "}\n"));
    write_file(si_dir, "matmul_4x8_8x2.onnx",
               text_to_serialized_model("<\n"
                                        "  ir_version: 7,\n"
                                        "  opset_import: [\"\" : 15]\n"
                                        ">\n"
                                        "agraph (float[4, 8] A, float[8, 2] B) => (float[4, 2] Y)\n"
                                        "{ Y = MatMul(A, B) }\n"));
    write_file(si_dir, "concat_axis0.onnx",
               text_to_serialized_model("<\n"
                                        "  ir_version: 7,\n"
                                        "  opset_import: [\"\" : 15]\n"
                                        ">\n"
                                        "agraph (float[2, 4] A, float[3, 4] B) => (float[5, 4] Y)\n"
                                        "{ Y = Concat<axis=0>(A, B) }\n"));
  } catch (const std::exception &ex) {
    std::cerr << "make_seed_corpus: " << ex.what() << "\n";
    return 1;
  }
  return 0;
}
