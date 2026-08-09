import pathlib
import unittest


class TestProtoVisibility(unittest.TestCase):
    def test_shared_proto_uses_explicit_visibility(self):
        root = pathlib.Path(__file__).resolve().parents[2]
        cmake = (root / "CMakeLists.txt").read_text(encoding="utf-8")
        visibility = (root / "onnx_light/onnx_proto/visibility.h").read_text(encoding="utf-8")
        stream_class = (root / "onnx_light/onnx_proto/stream_class.h").read_text(encoding="utf-8")
        onnx_pb = (root / "onnx_light/onnx_lib/common/onnx_pb.h").read_text(encoding="utf-8")

        self.assertIn("CXX_VISIBILITY_PRESET hidden", cmake)
        self.assertIn("VISIBILITY_INLINES_HIDDEN YES", cmake)
        self.assertIn(
            '#define ONNX_LIGHT_PROTO_API __attribute__((visibility("default")))', visibility
        )
        self.assertIn(
            "class ONNX_LIGHT_PROTO_API cls : public ProtoMessageAdapter<cls>", stream_class
        )
        self.assertIn("class ProtoMessageAdapter : public Message", stream_class)
        self.assertIn("#define ONNX_API ONNX_LIGHT_PROTO_API", onnx_pb)


if __name__ == "__main__":
    unittest.main(verbosity=2)
