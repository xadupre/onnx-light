from pathlib import Path


def test_python_abi_mode_is_configurable():
    root = Path(__file__).resolve().parents[2]
    cmake = (root / "CMakeLists.txt").read_text(encoding="utf-8")

    assert (
        "option(ONNX_LIGHT_PYTHON_STABLE_ABI\n"
        '    "Build Python extensions with CPython\'s stable ABI." ON)'
    ) in cmake
    assert "if(ONNX_LIGHT_PYTHON_STABLE_ABI)" in cmake
    assert "list(APPEND _onnx_light_nanobind_abi STABLE_ABI)" in cmake

    modules = {
        "_onnxpyprotoop": "ONNX_PY_PROTOOP_SOURCES",
        "_onnxpyprotolib": "ONNX_PY_PROTOLIB_SOURCES",
        "_onnxpycore": "ONNX_PY_CORE_SOURCES",
        "_onnxpypatterns": "ONNX_PY_PATTERNS_SOURCES",
        "_onnxpykernels": "ONNX_PY_KERNELS_SOURCES",
        "_onnxpybackend": "ONNX_PY_BACKEND_SOURCES",
        "_onnxpygradient": "ONNX_PY_GRADIENT_SOURCES",
    }
    for module, sources in modules.items():
        assert (
            f"nanobind_add_module({module} ${{_onnx_light_nanobind_abi}} ${{{sources}}})"
        ) in cmake
        assert f"nanobind_add_module({module} STABLE_ABI " not in cmake

    assert "if(MSVC AND ONNX_LIGHT_PYTHON_STABLE_ABI)" in cmake
    link_option = '"/NODEFAULTLIB:python${Python_VERSION_MAJOR}${Python_VERSION_MINOR}.lib"'
    assert link_option in cmake
