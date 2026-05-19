from ..onnx_py import _onnxpy

__all__ = [name for name in dir(_onnxpy) if not name.startswith("_")]

for name in __all__:
    globals()[name] = getattr(_onnxpy, name)
