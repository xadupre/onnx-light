import os

import onnx
from onnx import numpy_helper
from onnx.backend.test.loader import load_model_tests

# The TIFF, WebP and JPEG2000 formats are intentionally not decoded by the
# lightweight ``ImageDecoder`` kernel, so their upstream reference cases are
# skipped to keep the generated registry in sync with the kernel behavior.
skipped_formats = ("tiff", "webp", "jpeg2k", "jpeg2000")


def load_case(test_case):
    """Returns ``(model, input_array, output_array)`` for a node test case.

    Recent ``onnx`` releases build node backend-test data on the fly and
    expose the model in memory via ``TestCase.model`` and the reference
    tensors via ``TestCase.data_sets`` (propagated from onnx/onnx#7959).
    Older releases materialise the data on disk under ``TestCase.model_dir``;
    both layouts are supported here.
    """
    if test_case.model is not None:
        inputs, outputs = test_case.data_sets[0]
        return (test_case.model, _to_array(inputs[0]), _to_array(outputs[0]))
    data_dir = os.path.join(test_case.model_dir, "test_data_set_0")
    return (
        onnx.load(os.path.join(test_case.model_dir, "model.onnx")),
        numpy_helper.to_array(onnx.load_tensor(os.path.join(data_dir, "input_0.pb"))),
        numpy_helper.to_array(onnx.load_tensor(os.path.join(data_dir, "output_0.pb"))),
    )


def _to_array(value):
    if isinstance(value, onnx.TensorProto):
        return numpy_helper.to_array(value)
    return value


cases = sorted(
    (
        tc
        for tc in load_model_tests(kind="node")
        if tc.name.startswith("test_image_decoder_")
        and not any(fmt in tc.name for fmt in skipped_formats)
    ),
    key=lambda tc: tc.name,
)


def emit_uint8_array(name, data):
    out = [f"const unsigned char {name}[{len(data)}] = {{"]
    line = "   "
    for b in data:
        s = f" {b},"
        if len(line) + len(s) > 96:
            out.append(line)
            line = "   "
        line += s
    out.append(line)
    out.append("};")
    return "\n".join(out)


src = [
    "// Copyright (c) ONNX Project Contributors",
    "//",
    "// SPDX-License-Identifier: Apache-2.0",
    "//",
    "// This file is auto-generated from the upstream ONNX node-level",
    "// backend test data (``onnx/backend/test/data/node/test_image_decoder_*``)",
    "// by ``onnx_light/onnx_extensions/backend_test/cases/image/"
    "_generate_image_decoder_cases.py``.",
    "// Each case carries the encoded input bytestream and the precomputed",
    "// expected ``(H, W, C)`` uint8 image as static byte arrays so the",
    "// backend test library does not need to depend on an image-decoding",
    "// library to register coverage for ``ai.onnx::ImageDecoder``.",
    "//",
    "// Do not edit by hand: re-run the generator script if upstream ONNX",
    "// refreshes the reference data.",
    "",
    '#include "onnx_extensions/backend_test/cases/image/include_image_cases.h"',
    '#include "onnx_core/backend_test/expect.h"',
    '#include "onnx_proto/onnx_helper.h"',
    "",
    "#include <cstdint>",
    "#include <string>",
    "#include <vector>",
    "",
    "namespace ONNX_LIGHT_NAMESPACE {",
    "namespace onnx_kernels {",
    "",
    "namespace {",
    "",
]

entries = []  # (case_name, pixel_format, in_name, in_size, out_name, h, w, c)

for tc in cases:
    d = tc.name
    model, inp, out = load_case(tc)
    node = model.graph.node[0]
    pixel_format = "RGB"
    for a in node.attribute:
        if a.name == "pixel_format":
            pixel_format = a.s.decode()
    assert inp.dtype.name == "uint8" and inp.ndim == 1
    assert out.dtype.name == "uint8" and out.ndim == 3
    h, w, c = out.shape
    stem = d[len("test_") :]  # image_decoder_decode_*
    case_name = "test_cc_" + stem
    in_name = "k_" + stem + "_in"
    out_name = "k_" + stem + "_out"
    src.append(emit_uint8_array(in_name, inp.tobytes()))
    src.append("")
    src.append(emit_uint8_array(out_name, out.tobytes()))
    src.append("")
    entries.append((case_name, pixel_format, in_name, inp.size, out_name, h, w, c))

src.append("// Opset id for ``ai.onnx::ImageDecoder`` (introduced at opset 20).")
src.append("constexpr int64_t kImageDecoderSinceVersion = 20;")
src.append("")
src.append("}  // namespace")
src.append("")
src.append("// ---------------------------------------------------------------------------")
src.append(
    "// ImageDecoder — mirrors the upstream ``onnx/backend/test/case/node/"
    "image_decoder.py`` reference cases."
)
src.append("//")
src.append("// One case is registered per (format, pixel_format) pair. For each case")
src.append("// the encoded input bytestream and the expected ``(H, W, C)`` uint8")
src.append("// image are stored as static byte arrays embedded at code-generation")
src.append("// time from the upstream ONNX node-level test data so the registry is")
src.append("// self-contained and does not require any image-decoding library at")
src.append("// build or run time.")
src.append("// ---------------------------------------------------------------------------")
src.append("void RegisterImageDecoderCases(std::vector<TestCase> &registry) {")
src.append("  const OpsetId opset = DefaultOpset(kImageDecoderSinceVersion);")
src.append("")
src.append("  struct Entry {")
src.append("    const char *name;")
src.append("    const char *pixel_format;")
src.append("    const unsigned char *encoded;")
src.append("    int64_t encoded_size;")
src.append("    const unsigned char *expected;")
src.append("    int64_t height;")
src.append("    int64_t width;")
src.append("    int64_t channels;")
src.append("  };")
src.append("")
src.append("  static const Entry kEntries[] = {")
for name, pf, in_name, in_size, out_name, h, w, c in entries:
    src.append(f'      {{"{name}", "{pf}", {in_name}, {in_size}, {out_name}, {h}, {w}, {c}}},')
src.append("  };")
src.append("")
src.append("  for (const Entry &e : kEntries) {")
src.append("    NodeProto node;")
src.append('    node.set_op_type("ImageDecoder");')
src.append('    node.add_input("data");')
src.append('    node.add_output("output");')
src.append('    AddAttribute<std::string>(node, "pixel_format", e.pixel_format);')
src.append("")
src.append("    std::vector<uint8_t> in_bytes(e.encoded, e.encoded + e.encoded_size);")
src.append('    Tensor in_tensor = Tensor::FromUint8("", {e.encoded_size}, in_bytes);')
src.append("")
src.append("    const int64_t out_count = e.height * e.width * e.channels;")
src.append("    std::vector<uint8_t> out_bytes(e.expected, e.expected + out_count);")
src.append("    Tensor out_tensor =")
src.append('        Tensor::FromUint8("", {e.height, e.width, e.channels}, out_bytes);')
src.append("")
src.append('    Expect(node, {in_tensor}, {out_tensor}, e.name, {opset}, "backend-test",')
src.append("           registry);")
src.append("  }")
src.append("}")
src.append("")
src.append("}  // namespace onnx_kernels")
src.append("}  // namespace ONNX_LIGHT_NAMESPACE")

content = "\n".join(src) + "\n"
with open("/tmp/gen/cases_image_decoder.cc", "w") as f:
    f.write(content)
print("Wrote", len(content), "bytes")
