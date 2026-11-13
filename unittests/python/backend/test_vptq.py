# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Checks native VPTQ against real pinned upstream Torch dequantization."""

import json
import unittest
from pathlib import Path

import ml_dtypes
import numpy

import onnx_light.onnx.numpy_helper as onh
from onnx_light import onnx
from onnx_light.ext_test_case import ExtTestCase, import_or_skip

vptq = import_or_skip("onnx_light.onnx_core.vptq")
quantization = import_or_skip("onnx_light.onnx_core.quantization")


def fixture_records():
    """Returns independently generated upstream tensors without importing Torch."""
    path = Path(__file__).with_name("data") / "vptq_native_v1.json"
    return json.loads(path.read_text(encoding="ascii"))


def fixture_array(record):
    """Returns an owned native-endian array from little-endian fixture bytes."""
    dtype = numpy.dtype(ml_dtypes.bfloat16 if record["dtype"] == "bfloat16" else record["dtype"])
    return (
        numpy.frombuffer(bytes.fromhex(record["raw_hex"]), dtype=dtype.newbyteorder("<"))
        .reshape(record["shape"])
        .astype(dtype, copy=True)
    )


def fixtures():
    """Returns checkpoint arrays, explicit constructor parameters, and references."""
    for record in fixture_records()["fixtures"]:
        yield (
            record["name"],
            {name: fixture_array(value) for name, value in record["tensors"].items()},
            record["parameters"],
            fixture_array(record["expected"]).astype(numpy.float32),
        )


def restored(value):
    """Returns an independent protobuf wire round trip."""
    result = onnx.EncodedValueProto()
    result.ParseFromString(value.SerializeToString())
    return result


class TestVptq(ExtTestCase):
    def test_upstream_provenance_and_packed_boundary_coverage(self):
        data = fixture_records()
        provenance = data["provenance"]
        self.assertEqual(provenance["revision"], vptq.UPSTREAM_REVISION)
        self.assertEqual(provenance["runtime"], "2.10.0+cpu")
        self.assertEqual(provenance["device"], "cpu")
        self.assertIn("VPTQLinear.dequantize", provenance["generation"])
        self.assertEqual(len(provenance["source_files"]), 2)
        for digest in provenance["source_files"].values():
            self.assertEqual(len(bytes.fromhex(digest)), 32)
        first = data["fixtures"][0]
        words = fixture_array(first["tensors"]["indices"])
        codes = fixture_array(first["codes"]["indices"])
        self.assertEqual(words.shape, (1, 2, 3))
        self.assertTrue(numpy.any(words < 0))
        self.assertEqual(int(words[0, 0, 0].view(numpy.uint32)), 0xA8743C67)
        self.assertEqual(int(words[0, 0, 1].view(numpy.uint32)), 0xD0F19CCE)
        self.assertEqual(set(codes.reshape(-1)), set(range(8)))
        # A three-bit index beginning at bit 30 crosses the signed word boundary.
        self.assertNotEqual(int(codes[0, 0, 10]), 0)
        # The index starting at bit 63 requires nonzero bits from the third word.
        self.assertEqual(int(codes[0, 0, 21]), 3)
        dtypes = {record["expected"]["dtype"] for record in data["fixtures"]}
        self.assertEqual(dtypes, {"float16", "bfloat16", "float32"})

    def test_upstream_weights_and_lossless_native_round_trip(self):
        for name, tensors, parameters, expected in fixtures():
            with self.subTest(name=name):
                encoded = vptq.import_vptq_weights(tensors, **parameters)
                self.assertEqual(encoded.struct_type.name, "onnx_light.vptq_weights.v1")
                value = restored(encoded)
                actual = vptq.reconstruct_vptq_weights(value)
                self.assertEqual(actual.dtype, numpy.float32)
                self.assertEqual(actual.shape, expected.shape)
                # Compares exact stage-rounded upstream results, not an adapter oracle.
                self.assertEqual(actual.tobytes(), expected.tobytes())
                generic = onh.to_array(quantization.dequantize_tensor_proto(value))
                self.assertEqual(generic.dtype, numpy.float32)
                self.assertEqual(generic.tobytes(), expected.tobytes())
                exported = vptq.export_vptq_weights(value)
                self.assertEqual(set(exported), set(tensors))
                for key, tensor in tensors.items():
                    actual_tensor = exported[key]
                    self.assertEqual(actual_tensor.dtype, tensor.dtype)
                    self.assertEqual(actual_tensor.shape, tensor.shape)
                    self.assertEqual(actual_tensor.tobytes(), tensor.tobytes())
                config = vptq.export_vptq_config(value)
                for key, parameter in parameters.items():
                    actual_parameter = config[key]
                    if isinstance(parameter, list):
                        actual_parameter = list(actual_parameter)
                    self.assertEqual(actual_parameter, parameter)
                self.assertEqual(config["upstream_revision"], vptq.UPSTREAM_REVISION)
                self.assertEqual(config["schema_version"], 1)
                self.assertEqual(
                    vptq.import_vptq_weights(exported, **config).SerializeToString(),
                    encoded.SerializeToString(),
                )
                for tensor in tensors.values():
                    tensor.fill(0)
                self.assertEqual(
                    vptq.reconstruct_vptq_weights(value).tobytes(), expected.tobytes()
                )

    def test_optional_bias_is_preserved_and_runtime_cache_is_rejected(self):
        _, tensors, parameters, expected = next(
            item for item in fixtures() if item[0] == "combined_float16_norm0_perm_tails"
        )
        tensors["bias"] = numpy.linspace(
            -0.5, 0.5, parameters["out_features"], dtype=numpy.float16
        )
        encoded = vptq.import_vptq_weights(tensors, **parameters)
        self.assertEqual(vptq.reconstruct_vptq_weights(encoded).tobytes(), expected.tobytes())
        exported = vptq.export_vptq_weights(encoded)
        self.assertEqual(exported["bias"].tobytes(), tensors["bias"].tobytes())
        # Upstream inverse_perm is a nonpersistent runtime cache, not checkpoint state.
        tensors["inverse_perm"] = numpy.argsort(tensors["perm"]).astype(numpy.int32)
        with self.assertRaises(ValueError):
            vptq.import_vptq_weights(tensors, **parameters)

    def test_noncontiguous_loaded_arrays_preserve_logical_bytes(self):
        _, tensors, parameters, expected = next(fixtures())
        strided = {}
        for key, tensor in tensors.items():
            storage = numpy.empty((*tensor.shape[:-1], tensor.shape[-1] * 2), dtype=tensor.dtype)
            view = storage[..., ::2]
            view[...] = tensor
            self.assertFalse(view.flags.c_contiguous)
            strided[key] = view
        encoded = vptq.import_vptq_weights(strided, **parameters)
        exported = vptq.export_vptq_weights(encoded)
        for key, tensor in tensors.items():
            self.assertEqual(exported[key].tobytes(), tensor.tobytes())
        self.assertEqual(vptq.reconstruct_vptq_weights(encoded).tobytes(), expected.tobytes())
        # All three-bit patterns address this eight-entry codebook, including
        # codes now read from the original padding word.
        strided["indices"] = tensors["indices"][:, :, ::-1]
        encoded = vptq.import_vptq_weights(strided, **parameters)
        self.assertEqual(
            vptq.export_vptq_weights(encoded)["indices"].tobytes(), strided["indices"].tobytes()
        )

    def test_explicit_constructor_metadata_is_not_inferred_from_shapes(self):
        _, tensors, parameters, expected = next(fixtures())
        parameters = {
            **parameters,
            "bits": 6,
            "kmeans_mode": "source-metadata",
            "kmeans_iters": 17,
            "kmeans_tol": 0.000123,
            "kmeans_seed": 13,
            "block_size": 37,
            "percdamp": 0.03125,
        }
        # bits is training metadata, not the packed codebook index width.
        encoded = vptq.import_vptq_weights(tensors, **parameters)
        config = vptq.export_vptq_config(encoded)
        for key in (
            "bits",
            "kmeans_mode",
            "kmeans_iters",
            "kmeans_tol",
            "kmeans_seed",
            "block_size",
            "percdamp",
        ):
            self.assertEqual(config[key], parameters[key])
        self.assertEqual(vptq.reconstruct_vptq_weights(encoded).tobytes(), expected.tobytes())
        for missing in ("in_features", "out_features"):
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                vptq.import_vptq_weights(
                    tensors, **{key: value for key, value in parameters.items() if key != missing}
                )

    def test_rejects_missing_enabled_side_tensors(self):
        for name, tensors, parameters, _ in fixtures():
            for missing in tensors:
                if missing == "bias":
                    continue
                with self.subTest(name=name, missing=missing), self.assertRaises(ValueError):
                    vptq.import_vptq_weights(
                        {key: value for key, value in tensors.items() if key != missing},
                        **parameters,
                    )

    def test_rejects_invalid_outlier_codes_and_permutations(self):
        _, tensors, parameters, _ = next(
            item for item in fixtures() if item[0] == "combined_float16_norm0_perm_tails"
        )
        for code in (-1, parameters["num_centroids"][0], 2**31 - 1):
            invalid = tensors["outlier_indices"].copy()
            invalid.flat[0] = code
            with self.subTest(code=code), self.assertRaises(ValueError):
                vptq.import_vptq_weights({**tensors, "outlier_indices": invalid}, **parameters)
        for perm in (
            numpy.zeros_like(tensors["perm"]),
            numpy.full_like(tensors["perm"], -1),
            numpy.arange(parameters["in_features"], dtype=numpy.int32) + 1,
            tensors["perm"][:-1].copy(),
            tensors["perm"].astype(numpy.int64),
        ):
            with self.subTest(perm=perm), self.assertRaises(ValueError):
                vptq.import_vptq_weights({**tensors, "perm": perm}, **parameters)

    def test_rejects_unpacked_external_variants_and_bad_arrays(self):
        _, tensors, parameters, _ = next(fixtures())
        record = fixture_records()["fixtures"][0]
        for changes in (
            {"indices": fixture_array(record["codes"]["indices"])},
            {"indices": tensors["indices"].astype(numpy.int64)},
            {"indices": tensors["indices"].reshape(-1)},
            {"indices": tensors["indices"][:, :, :-1].copy()},
            {"indices": tensors["indices"].astype(">i4")},
            {"indices": tensors["indices"].tobytes()},
            {"centroids.weight": tensors["centroids.weight"].reshape(1, 8, 3)},
            {"centroids.weight": tensors["centroids.weight"][:, :-1].copy()},
            {"centroids.weight": tensors["centroids.weight"].astype(numpy.float64)},
            {"qweight": tensors["indices"]},
            {"res_indices": numpy.zeros_like(tensors["indices"])},
            {"weight_scale": numpy.ones(parameters["in_features"], dtype=numpy.float32)},
            {"perm": numpy.arange(parameters["in_features"], dtype=numpy.int32)},
        ):
            with self.subTest(changes=list(changes)), self.assertRaises((ValueError, TypeError)):
                vptq.import_vptq_weights({**tensors, **changes}, **parameters)

    def test_rejects_non_power_of_two_codebook_overflow(self):
        _, tensors, parameters, _ = next(fixtures())
        # Seven entries still occupy three bits, but code seven is not addressable.
        with self.assertRaises(ValueError):
            vptq.import_vptq_weights(
                {**tensors, "centroids.weight": tensors["centroids.weight"][:, :21].copy()},
                **{**parameters, "num_centroids": (-1, 7)},
            )
        _, tensors, parameters, _ = next(
            item for item in fixtures() if item[0] == "residual_float16_cross_word"
        )
        with self.assertRaises(ValueError):
            vptq.import_vptq_weights(
                {
                    **tensors,
                    "res_centroids.weight": tensors["res_centroids.weight"][:, :9].copy(),
                },
                **{**parameters, "num_res_centroids": (-1, 3)},
            )

    def test_rejects_mismatched_side_tensor_dtypes_and_shapes(self):
        _, tensors, parameters, _ = next(
            item for item in fixtures() if item[0] == "combined_float16_norm0_perm_tails"
        )
        for key in (
            "res_centroids.weight",
            "outlier_centroids.weight",
            "weight_scale",
            "weight_bias",
        ):
            for value in (
                tensors[key].astype(numpy.float32),
                tensors[key].reshape(-1)[:-1].copy(),
            ):
                with self.subTest(key=key, shape=value.shape), self.assertRaises(ValueError):
                    vptq.import_vptq_weights({**tensors, key: value}, **parameters)
        with self.assertRaises(ValueError):
            vptq.import_vptq_weights(
                {**tensors, "bias": numpy.zeros(parameters["out_features"], dtype=numpy.float32)},
                **parameters,
            )

    def test_rejects_geometry_revision_version_and_resource_limits(self):
        _, tensors, parameters, _ = next(fixtures())
        for changes in (
            {"in_features": 0},
            {"out_features": -1},
            {"in_features": 2**62, "out_features": 2**62},
            {"in_features": True},
            {"in_features": 23.0},
            {"bits": 0},
            {"bits": 9},
            {"vector_lens": (-1, 0)},
            {"vector_lens": (-1, 2**62)},
            {"vector_lens": (3,)},
            {"vector_lens": (-1, 3.0)},
            {"num_centroids": (-1, 1)},
            {"num_centroids": (-1, 7)},
            {"num_centroids": (-1, 2**62)},
            {"num_res_centroids": (-1, 1)},
            {"num_res_centroids": (-1, 3)},
            {"group_size": 0},
            {"group_size": 2, "group_num": 2},
            {"group_num": parameters["in_features"] + 1},
            {"npercent": 1.0},
            {"npercent": numpy.nan},
            {"npercent": numpy.inf},
            {"norm_dim": 2},
            {"enable_perm": 1},
            {"enable_norm": "false"},
            {"upstream_revision": "main"},
            {"schema_version": 0},
            {"schema_version": 2},
            {"schema_version": True},
            {"max_bytes": 0},
            {"max_bytes": 1},
            {"max_bytes": parameters["in_features"] * parameters["out_features"] * 4 - 1},
            {"variant": "microsoft-vptq-unpacked"},
            {"layout": "external-vptq"},
        ):
            with self.subTest(changes=changes), self.assertRaises((ValueError, TypeError)):
                vptq.import_vptq_weights(tensors, **{**parameters, **changes})
        encoded = vptq.import_vptq_weights(tensors, **parameters)
        for consumer in (
            vptq.export_vptq_weights,
            vptq.export_vptq_config,
            vptq.reconstruct_vptq_weights,
        ):
            with self.subTest(consumer=consumer), self.assertRaises(ValueError):
                consumer(encoded, max_bytes=1)

    def test_checkpoint_storage_limit_is_independent_of_logical_output(self):
        _, tensors, parameters, _ = next(fixtures())
        parameters = {**parameters, "num_centroids": (-1, 256)}
        tensors = {
            "centroids.weight": numpy.resize(tensors["centroids.weight"], (1, 256 * 3)),
            "indices": numpy.zeros((1, 2, 6), dtype=numpy.int32),
        }
        storage_bytes = sum(tensor.nbytes for tensor in tensors.values())
        logical_bytes = parameters["in_features"] * parameters["out_features"] * 4
        self.assertGreater(storage_bytes - 1, logical_bytes)
        with self.assertRaises(ValueError):
            vptq.import_vptq_weights(tensors, **parameters, max_bytes=storage_bytes - 1)
        encoded = vptq.import_vptq_weights(tensors, **parameters, max_bytes=storage_bytes)
        for consumer in (
            vptq.export_vptq_weights,
            vptq.export_vptq_config,
            vptq.reconstruct_vptq_weights,
        ):
            with self.subTest(consumer=consumer):
                consumer(encoded, max_bytes=storage_bytes)
                with self.assertRaises(ValueError):
                    consumer(encoded, max_bytes=storage_bytes - 1)

    def test_tiny_checkpoint_has_matching_producer_consumer_byte_limits(self):
        parameters = dict(
            in_features=1, out_features=1, vector_lens=(-1, 1), num_centroids=(-1, 2)
        )
        tensors = {
            "indices": numpy.array([[[1]]], dtype=numpy.int32),
            "centroids.weight": numpy.array([[-0.125, 1.25]], dtype=numpy.float32),
        }
        self.assertEqual(sum(tensor.nbytes for tensor in tensors.values()), 12)
        encoded = vptq.import_vptq_weights(tensors, **parameters, max_bytes=12)
        exported = vptq.export_vptq_weights(encoded, max_bytes=12)
        for key, tensor in tensors.items():
            self.assertEqual(exported[key].tobytes(), tensor.tobytes())
        config = vptq.export_vptq_config(encoded, max_bytes=12)
        self.assertEqual(
            vptq.import_vptq_weights(exported, **config, max_bytes=12).SerializeToString(),
            encoded.SerializeToString(),
        )
        numpy.testing.assert_array_equal(
            vptq.reconstruct_vptq_weights(encoded, max_bytes=12),
            numpy.array([[1.25]], dtype=numpy.float32),
        )
        with self.assertRaises(ValueError):
            vptq.import_vptq_weights(tensors, **parameters, max_bytes=11)
        for consumer in (
            vptq.export_vptq_weights,
            vptq.export_vptq_config,
            vptq.reconstruct_vptq_weights,
        ):
            with self.subTest(consumer=consumer), self.assertRaises(ValueError):
                consumer(encoded, max_bytes=11)

    def test_rejects_mutated_payload_and_schema(self):
        _, tensors, parameters, _ = next(fixtures())
        encoded = vptq.import_vptq_weights(tensors, **parameters)
        invalid = []
        for raw in (encoded.raw_data[:-1], encoded.raw_data + b"\0"):
            value = restored(encoded)
            value.raw_data = raw
            invalid.append(value)
        value = restored(encoded)
        value.struct_type.name = "onnx_light.vptq_weights.v2"
        invalid.append(value)
        value = restored(encoded)
        value.logical_type.tensor_type.shape.dim[0].dim_value += 1
        invalid.append(value)
        value = restored(encoded)
        value.logical_type.tensor_type.elem_type = onnx.TensorProto.FLOAT16
        invalid.append(value)
        for field, integer in (
            ("packed", 0),
            ("pack_dtype", onnx.TensorProto.INT64),
            ("in_features", 2**62),
            ("out_features", 2**62),
            ("num_centroids", 2**62),
            ("vector_len", 2**62),
            ("group_size", 0),
        ):
            value = restored(encoded)
            entry = next(item for item in value.struct_type.structure.field if item.name == field)
            entry.constant.int64_data.clear()
            entry.constant.int64_data.append(integer)
            invalid.append(value)
        value = restored(encoded)
        revision = next(
            item
            for item in value.struct_type.structure.field
            if item.name == "gptqmodel_revision"
        )
        revision.constant.raw_data = b"main"
        invalid.append(value)
        for value in invalid:
            for consumer in (
                vptq.export_vptq_weights,
                vptq.export_vptq_config,
                vptq.reconstruct_vptq_weights,
            ):
                with self.subTest(consumer=consumer), self.assertRaises(ValueError):
                    consumer(value)


if __name__ == "__main__":
    unittest.main()
