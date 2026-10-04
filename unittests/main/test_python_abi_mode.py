from pathlib import Path


def test_python_abi_mode_is_configurable():
    root = Path(__file__).resolve().parents[2]
    cmake = (root / "CMakeLists.txt").read_text(encoding="utf-8")

    assert "option(ONNX_LIGHT_PYTHON_STABLE_ABI" in cmake
    assert "if(ONNX_LIGHT_PYTHON_STABLE_ABI)" in cmake
    assert "list(APPEND _onnx_light_nanobind_abi STABLE_ABI)" in cmake
    assert (
        "nanobind_add_module(_onnxpykernels ${_onnx_light_nanobind_abi} "
        "${ONNX_PY_KERNELS_SOURCES})"
    ) in cmake
