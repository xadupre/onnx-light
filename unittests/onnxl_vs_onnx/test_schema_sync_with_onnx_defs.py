import unittest
from onnx_light.ext_test_case import ExtTestCase
import onnx
import onnx.defs as onnx_defs
import onnx_light.onnx.defs


class TestSchemaSyncWithOnnxDefs(ExtTestCase):
    # SwiGLU (onnx#8202) may not yet be present in the installed onnx; the empty
    # domain string is the default ai.onnx domain used by the registered schemas.
    SWIGLU_KEY = ("", "SwiGLU", 28)
    BITSHIFT_28_KEY = ("", "BitShift", 28)
    OPTIONAL_28_KEYS = {
        ("", "Optional", 28),
        ("", "OptionalGetElement", 28),
        ("", "OptionalHasElement", 28),
    }
    REDUCE_LOG_28_KEYS = {("", "ReduceLogSum", 28), ("", "ReduceLogSumExp", 28)}
    DATA_LAYOUT_28_KEYS = {
        ("", name, 28)
        for name in (
            "Compress",
            "DepthToSpace",
            "Einsum",
            "Mod",
            "OneHot",
            "ReverseSequence",
            "SpaceToDepth",
            "Unique",
        )
    }
    ATTENTION_25_KEY = ("", "Attention", 25)
    GEMM_29_KEY = ("", "Gemm", 29)
    DEQUANTIZE_PACKED_29_KEY = ("", "DequantizePacked", 29)
    ND_29_KEYS = {("", "GatherND", 29), ("", "ScatterND", 29)}
    STFT_KEY = ("", "STFT", 17)
    STFT_DOC_MARKER = "frames = floor((signal_length - frame_length) / frame_step) + 1"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        onnx_light.onnx.defs.register_onnx_operator_set_schema()

    @classmethod
    def _remove_stft_if_onnx_docs_are_outdated(cls, light_dict, onnx_dict):
        if cls.STFT_DOC_MARKER not in onnx_dict[cls.STFT_KEY].doc:
            light_dict.pop(cls.STFT_KEY)
            onnx_dict.pop(cls.STFT_KEY)

    @classmethod
    def _remove_recent_schemas_if_onnx_is_outdated(cls, light_dict, onnx_dict):
        recent_keys = (
            cls.OPTIONAL_28_KEYS
            | cls.REDUCE_LOG_28_KEYS
            | cls.DATA_LAYOUT_28_KEYS
            | {cls.GEMM_29_KEY}
            | {cls.DEQUANTIZE_PACKED_29_KEY}
            | cls.ND_29_KEYS
        )
        for key in recent_keys - set(onnx_dict):
            if isinstance(light_dict, set):
                light_dict.discard(key)
            else:
                light_dict.pop(key, None)

    @staticmethod
    def _normalize_div_doc(key, doc):
        if key in {("", "Div", version) for version in (6, 7, 13, 14)}:
            # Installed ONNX may predate the clarification in onnx#8461.
            doc = doc.replace(
                "For integer inputs, the result is computed using truncating division "
                "(rounding toward zero).\n",
                "For integer inputs, the result is computed using truncating division "
                "(rounding toward zero). For example, `-11 / 3` yields `-3`.\n",
            )
        return doc

    @staticmethod
    def _normalize_nd_text(key, text):
        # ONNX wheels may predate onnx#8692's edits to historical descriptions.
        # Normalize only those exact edits, retaining all other schema comparisons.
        if key == ("", "GatherND", 13):
            replacements = {
                (
                    "The number of batch dimensions. The gather of indexing starts "
                    "from dimension of data[batch_dims:]"
                ): (
                    "The number of leading batch dimensions shared by data and indices. "
                    "Index tuples address data axes starting at axis batch_dims."
                ),
                (
                    "Tensor of rank q >= 1. All index values are expected to be within "
                    "bounds [-s, s-1] along axis of size s. It is an error if any of "
                    "the index values are out of bounds."
                ): (
                    "Tensor of rank q >= 1. Each index must be in [-s, s-1], where s is "
                    "the size of the corresponding data axis after the batch dimensions. "
                    "An out-of-bounds index is an error."
                ),
                "Tensor of rank q + r - indices_shape[-1] - 1.": (
                    "Tensor of rank q + r - indices_shape[-1] - 1 - batch_dims."
                ),
                (
                    "`indices` is an q-dimensional integer tensor, best thought of as a "
                    "`(q-1)`-dimensional tensor of index-tuples into `data`,\n"
                    "where each element defines a slice of `data`\n\n"
                    "`batch_dims` (denoted as `b`) is an integer indicating the number of batch "
                    "dimensions, i.e the leading `b` number of dimensions of\n"
                    "`data` tensor and `indices` are representing the batches, and the gather "
                    "starts from the `b+1` dimension.\n\n"
                    "Some salient points about the inputs' rank and shape:\n\n"
                    "1) r >= 1 and q >= 1 are to be honored. There is no dependency condition "
                    "to be met between ranks `r` and `q`\n\n"
                    "2) The first `b` dimensions of the shape of `indices` tensor and `data` "
                    "tensor must be equal.\n\n"
                    "3) b < min(q, r) is to be honored.\n\n"
                    "4) The `indices_shape[-1]` should have a value between 1 (inclusive) "
                    "and rank `r-b` (inclusive)\n\n"
                    "5) All values in `indices` are expected to be within bounds [-s, s-1] "
                    "along axis of size `s` (i.e.) "
                    "`-data_shape[i] <= indices[...,i] <= data_shape[i] - 1`.\n"
                    "   It is an error if any of the index values are out of bounds.\n\n"
                    "The output is computed as follows:\n\n"
                    "The output tensor is obtained by mapping each index-tuple in the "
                    "`indices` tensor to the corresponding slice of the input `data`.\n\n"
                    "1) If `indices_shape[-1] > r-b` => error condition\n\n"
                    "2) If `indices_shape[-1] == r-b`, since the rank of `indices` is `q`, "
                    "`indices` can be thought of as `N` `(q-b-1)`-dimensional tensors\n"
                    "   containing 1-D tensors of dimension `r-b`, where `N` is an integer "
                    "equals to the product of 1 and all the elements in the batch dimensions\n"
                    "   of the indices_shape. Let us think of each such `r-b` ranked tensor "
                    "as `indices_slice`. Each *scalar value* corresponding to "
                    "`data[0:b-1,indices_slice]`\n"
                    "   is filled into the corresponding location of the `(q-b-1)`-dimensional "
                    "tensor to form the `output` tensor (Example 1 below)\n\n"
                    "3) If `indices_shape[-1] < r-b`, since the rank of `indices` is `q`, "
                    "`indices` can be thought of as `N` `(q-b-1)`-dimensional tensor\n"
                    "   containing 1-D tensors of dimension `< r-b`. Let us think of each "
                    "such tensors as `indices_slice`. Each *tensor slice* corresponding\n"
                    "   to `data[0:b-1, indices_slice , :]` is filled into the corresponding "
                    "location of the `(q-b-1)`-dimensional tensor\n"
                    "   to form the `output` tensor (Examples 2, 3, 4 and 5 below)\n\n"
                    "This operator is the inverse of `ScatterND`."
                ): (
                    "Let `k = indices_shape[-1]`. Each length-`k` tuple in `indices` "
                    "selects an element or slice of `data`.\n\n"
                    "The leading `b` dimensions of `data` and `indices` are batch dimensions. "
                    "Each tuple indexes\n"
                    "the next `k` dimensions of `data`, starting at axis `b` "
                    "(axes are zero-based).\n\n"
                    "The inputs must satisfy the following constraints:\n\n"
                    "1) `r >= 1` and `q >= 1`.\n\n"
                    "2) The first `b` dimensions of the shape of `indices` tensor and `data` "
                    "tensor must be equal.\n\n"
                    "3) `batch_dims` must satisfy `0 <= b < min(q, r)`.\n\n"
                    "4) `indices_shape[-1]` must be between 1 and `r-b`, inclusive.\n\n"
                    "5) For each tuple component `i`, `0 <= i < indices_shape[-1]`, "
                    "the index must satisfy\n"
                    "   `-data_shape[b+i] <= indices[...,i] < data_shape[b+i]`.\n"
                    "   It is an error if any of the index values are out of bounds.\n\n"
                    "The output shape is `indices_shape[:-1] + data_shape[b+k:]`, "
                    "where `+` denotes shape concatenation.\n"
                    "For each coordinate tuple `p` in `indices_shape[:-1]`, "
                    "the selected value or slice is\n"
                    "`output[p] = data[p[:b] + tuple(indices[p])]`, "
                    "using tuple concatenation for the data coordinate.\n"
                    "Unspecified trailing data axes select the entire slice. "
                    "When `k = r-b`, each selected value is a scalar.\n"
                    "Negative indices count backwards from the end of the "
                    "corresponding data dimension.\n"
                    "Within the same batch, repeated index tuples select the same value "
                    "or slice at multiple output positions."
                ),
            }
        elif key == ("", "ScatterND", 18):
            replacements = {
                "'add':  reduction using the addition operation. ": (
                    "'add': reduction using the addition operation. "
                ),
                "'mul':  reduction using the addition operation. ": (
                    "'mul': reduction using the multiplication operation. "
                ),
                "'max': reduction using the maximum operation.'min'": (
                    "'max': reduction using the maximum operation. 'min'"
                ),
                (
                    "update to a slice of the tensor. Index values are allowed to be "
                    "negative, as per the usual\n"
                    "convention for counting backwards from the end, but are expected "
                    "in the valid range."
                ): (
                    "update to a slice of the tensor. For each tuple component `i`, "
                    "`0 <= i < k`, the index must satisfy\n"
                    "`-data.shape[i] <= indices[...,i] < data.shape[i]`. Negative indices "
                    "count backwards from the end\n"
                    "of the corresponding dimension. An out-of-bounds index is an error."
                ),
                (
                    "In particular, indices should not have duplicate entries: that is, "
                    "if idx1 != idx2, then indices[idx1] != indices[idx2].\n"
                    "This ensures that the output value does not depend on the iteration order.\n"
                ): "",
                (
                    'In cases where `reduction` is set to "none", indices should not have '
                    "duplicate entries: that is, if idx1 != idx2,\n"
                    "then indices[idx1] != indices[idx2]. This ensures that the output value "
                    "does not depend on the iteration order."
                ): (
                    'When `reduction` is "none", no two index tuples may select the same '
                    "output element or slice,\n"
                    "including through equivalent positive and negative indices. "
                    "This ensures that the output value does not\n"
                    "depend on the iteration order."
                ),
                "This operator is the inverse of GatherND.\n\n": "",
            }
        else:
            return text
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def test_normalize_div_doc(self):
        example = " For example, `-11 / 3` yields `-3`."
        for version in (6, 7, 13, 14):
            with self.subTest(version=version):
                key = ("", "Div", version)
                current_doc = onnx_light.onnx.defs.get_schema("Div", version).doc
                self.assertIn(example, current_doc)
                old_doc = current_doc.replace(example, "")
                self.assertEqual(self._normalize_div_doc(key, old_doc), current_doc)
                self.assertEqual(self._normalize_div_doc(key, current_doc), current_doc)
                for other_key in (
                    ("", "Div", 1),
                    ("", "Add", version),
                    ("custom", "Div", version),
                ):
                    self.assertEqual(self._normalize_div_doc(other_key, old_doc), old_doc)

    def test_recent_nd_schemas_are_compared_when_available(self):
        historical = {("", "GatherND", 13), ("", "ScatterND", 18)}
        unrelated = {("custom", "GatherND", 29), ("", "GatherND", 30)}
        keys = self.ND_29_KEYS | historical | unrelated
        for use_set in (False, True):
            for available in (set(), {("", "GatherND", 29)}, self.ND_29_KEYS):
                with self.subTest(use_set=use_set, available=available):
                    light = set(keys) if use_set else dict.fromkeys(keys)
                    upstream = available | historical
                    self._remove_recent_schemas_if_onnx_is_outdated(light, upstream)
                    self.assertEqual(set(light), available | historical | unrelated)

    def test_normalize_nd_text(self):
        for name, version in (("GatherND", 13), ("ScatterND", 18)):
            key = ("", name, version)
            upstream = onnx_defs.get_schema(name, version)
            light = onnx_light.onnx.defs.get_schema(name, version)
            for old, new in (
                [(upstream.doc, light.doc)]
                + [(a.description, b.description) for a, b in zip(upstream.inputs, light.inputs)]
                + [
                    (a.description, b.description)
                    for a, b in zip(upstream.outputs, light.outputs)
                ]
                + [
                    (upstream.attributes[n].description, light.attributes[n].description)
                    for n in upstream.attributes
                ]
            ):
                with self.subTest(key=key):
                    self.assertEqual(self._normalize_nd_text(key, old), new)
                    self.assertEqual(self._normalize_nd_text(key, new), new)
                    self.assertEqual(
                        self._normalize_nd_text(key, old + "\nother text"), new + "\nother text"
                    )
                    self.assertEqual(self._normalize_nd_text(("custom", name, version), old), old)
                    self.assertEqual(self._normalize_nd_text(("", name, 29), old), old)

    def test_onnx_light_ir_and_opset_versions_match_onnx(self):
        self.assertEqual(onnx_light.onnx.defs.onnx_ir_version(), onnx.IR_VERSION)
        self.assertGreaterEqual(
            onnx_light.onnx.defs.onnx_opset_version(), onnx_defs.onnx_opset_version()
        )

    def test_registered_onnx_ops_match_onnx_registered(self):
        flex_attention_key = ("ai.onnx.preview", "FlexAttention", 1)
        onnx_light_schema_keys = {
            (schema.domain, schema.name, schema.since_version)
            for schema in onnx_light.onnx.defs.get_all_schemas_with_history()
        }
        onnx_schema_keys = {
            (schema.domain, schema.name, schema.since_version)
            for schema in onnx_defs.get_all_schemas_with_history()
        }

        if flex_attention_key not in onnx_schema_keys:
            onnx_light_schema_keys.discard(flex_attention_key)
        if self.SWIGLU_KEY not in onnx_schema_keys:
            onnx_light_schema_keys.discard(self.SWIGLU_KEY)
        if self.BITSHIFT_28_KEY not in onnx_schema_keys:
            onnx_light_schema_keys.discard(self.BITSHIFT_28_KEY)
        self._remove_recent_schemas_if_onnx_is_outdated(onnx_light_schema_keys, onnx_schema_keys)
        if self.ATTENTION_25_KEY not in onnx_schema_keys:
            onnx_light_schema_keys.discard(self.ATTENTION_25_KEY)

        self.assertEqual(onnx_light_schema_keys, onnx_schema_keys)

    def test_registered_onnx_ops_match_onnx_match_input_output(self):
        light_hist = onnx_light.onnx.defs.get_all_schemas_with_history()
        hist = onnx.defs.get_all_schemas_with_history()
        light_dict = {(s.domain, s.name, s.since_version): s for s in light_hist}
        onnx_dict = {(s.domain, s.name, s.since_version): s for s in hist}
        if self.SWIGLU_KEY not in onnx_dict:
            light_dict.pop(self.SWIGLU_KEY, None)
        if self.BITSHIFT_28_KEY not in onnx_dict:
            light_dict.pop(self.BITSHIFT_28_KEY, None)
        self._remove_recent_schemas_if_onnx_is_outdated(light_dict, onnx_dict)
        if self.ATTENTION_25_KEY not in onnx_dict:
            light_dict.pop(self.ATTENTION_25_KEY, None)
        self._remove_stft_if_onnx_docs_are_outdated(light_dict, onnx_dict)
        self.assertEqual(set(light_dict), set(onnx_dict))
        for key, lights in light_dict.items():
            with self.subTest(key=key):
                schema = onnx_dict[key]
                self.assertGreater(len(schema.outputs), 0)
                self.assertGreater(len(lights.outputs), 0)
                self.assertGreater(schema.line, 0)
                self.assertGreater(lights.line, 0)
                self.assertEqual(len(schema.inputs), len(lights.inputs))
                self.assertEqual(len(schema.outputs), len(lights.outputs))
                self.assertEqual(len(schema.type_constraints), len(lights.type_constraints))
                if key == ("", "MeanVarianceNormalization", 13):
                    # Correct FLOAT16/DOUBLE expansion needs the input dtype.
                    self.assertFalse(lights.has_function)
                    self.assertTrue(lights.has_context_dependent_function)
                    self.assertEqual(lights.all_function_opset_versions, [13, 18])
                else:
                    self.assertEqual(schema.has_function, lights.has_function)
                self.assertEqual(schema.deprecated, lights.deprecated)
                self.assertEqual(schema.is_infinite(0), lights.is_infinite(0))
                self.assertEqual(schema.max_input, lights.max_input)
                self.assertEqual(schema.max_output, lights.max_output)
                # file and line are build-environment-specific and intentionally not compared
                self.assertEqual(schema.non_deterministic, lights.non_deterministic)
                self.assertEqual(schema.support_level.name, lights.support_level.name)
                onnx_doc = self._normalize_nd_text(key, self._normalize_div_doc(key, schema.doc))
                if key == ("", "GroupNormalization", 21):
                    # Installed ONNX may predate the documentation fix in onnx#8428.
                    onnx_doc = onnx_doc.replace(
                        "`scale` and `bias` should be specified for each channel. The number of\n"
                        "groups `num_groups` should be divisible by the number of channels "
                        "so that there are\n"
                        "an equal number of channels per group.",
                        "`scale` and `bias` should be specified for each channel. "
                        "The number of channels\n"
                        "should be divisible by `num_groups` so that there are "
                        "an equal number of channels\n"
                        "per group.",
                    )
                if key == ("ai.onnx.ml", "Normalizer", 1):
                    # Installed ONNX may predate the documentation fix in onnx#8451.
                    onnx_doc = onnx_doc.replace(
                        "Normalize the input.  There are three normalization modes, "
                        "which have the corresponding formulas,\n"
                        "    defined using element-wise infix operators '/' and '^' "
                        "and tensor-wide functions 'max' and 'sum':<br>",
                        "Normalize the input. There are three normalization modes, "
                        "which have the corresponding formulas.\n"
                        "    The function 'abs' and operator '^' are element-wise, "
                        "while 'max' and 'sum' reduce along the normalization axis:<br>",
                    )
                    onnx_doc = onnx_doc.replace("Y = X / sum(X)", "Y = X / sum(abs(X))")
                    onnx_doc = onnx_doc.replace(
                        "Y = sqrt(X^2 / sum(X^2)}", "Y = X / sqrt(sum(X^2))"
                    )
                self.assertEqual(onnx_doc, lights.doc)

    def test_registered_onnx_ops_match_onnx_match_input_output_doc(self):
        light_hist = onnx_light.onnx.defs.get_all_schemas_with_history()
        hist = onnx.defs.get_all_schemas_with_history()
        light_dict = {(s.domain, s.name, s.since_version): s for s in light_hist}
        onnx_dict = {(s.domain, s.name, s.since_version): s for s in hist}
        if self.SWIGLU_KEY not in onnx_dict:
            light_dict.pop(self.SWIGLU_KEY, None)
        if self.BITSHIFT_28_KEY not in onnx_dict:
            light_dict.pop(self.BITSHIFT_28_KEY, None)
        self._remove_recent_schemas_if_onnx_is_outdated(light_dict, onnx_dict)
        if self.ATTENTION_25_KEY not in onnx_dict:
            light_dict.pop(self.ATTENTION_25_KEY, None)
        self._remove_stft_if_onnx_docs_are_outdated(light_dict, onnx_dict)
        self.assertEqual(set(light_dict), set(onnx_dict))

        def _normalize(text):
            # Whitespace differences (spaces, tabs, newlines) are tolerated.
            return " ".join((text or "").split())

        for key, lights in light_dict.items():
            schema = onnx_dict[key]
            with self.subTest(key=key):
                for index, (onnx_in, light_in) in enumerate(zip(schema.inputs, lights.inputs)):
                    with self.subTest(input=index, name=onnx_in.name):
                        self.assertEqual(onnx_in.name, light_in.name)
                        self.assertEqual(
                            _normalize(self._normalize_nd_text(key, onnx_in.description)),
                            _normalize(light_in.description),
                        )
                for index, (onnx_out, light_out) in enumerate(
                    zip(schema.outputs, lights.outputs)
                ):
                    with self.subTest(output=index, name=onnx_out.name):
                        self.assertEqual(onnx_out.name, light_out.name)
                        self.assertEqual(
                            _normalize(self._normalize_nd_text(key, onnx_out.description)),
                            _normalize(light_out.description),
                        )

    def test_registered_onnx_ops_match_onnx_match_attributes(self):
        light_hist = onnx_light.onnx.defs.get_all_schemas_with_history()
        hist = onnx.defs.get_all_schemas_with_history()
        light_dict = {(s.domain, s.name, s.since_version): s for s in light_hist}
        onnx_dict = {(s.domain, s.name, s.since_version): s for s in hist}
        if self.SWIGLU_KEY not in onnx_dict:
            light_dict.pop(self.SWIGLU_KEY, None)
        if self.BITSHIFT_28_KEY not in onnx_dict:
            light_dict.pop(self.BITSHIFT_28_KEY, None)
        self._remove_recent_schemas_if_onnx_is_outdated(light_dict, onnx_dict)
        if self.ATTENTION_25_KEY not in onnx_dict:
            light_dict.pop(self.ATTENTION_25_KEY, None)
        self._remove_stft_if_onnx_docs_are_outdated(light_dict, onnx_dict)
        self.assertEqual(set(light_dict), set(onnx_dict))
        for key, lights in light_dict.items():
            schema = onnx_dict[key]
            with self.subTest(key=key):
                self.assertEqual(set(schema.attributes), set(lights.attributes))
                for name in schema.attributes:
                    onnx_attr = schema.attributes[name]
                    light_attr = lights.attributes[name]
                    with self.subTest(attribute=name):
                        self.assertEqual(onnx_attr.name, light_attr.name)
                        self.assertEqual(onnx_attr.required, light_attr.required)
                        # AttrType in onnx and AttributeType in onnx_light share enum
                        # names ("INT", "FLOAT", ...).
                        self.assertEqual(onnx_attr.type.name, light_attr.type.name)
                        self.assertEqual(
                            self._normalize_nd_text(key, onnx_attr.description),
                            light_attr.description,
                        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
