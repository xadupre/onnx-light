---
name: import-onnx-helper
description: Uses the repository-standard ONNX helper module import. Use when adding or editing Python code that calls onnx_light.onnx.helper.
---

# Import the ONNX helper module

When Python code uses `onnx_light.onnx.helper`, import the module directly with
the standard short alias:

```python
import onnx_light.onnx.helper as oh
```

Call its functions through `oh`, for example:

```python
node = oh.make_node("Identity", ["X"], ["Y"])
```

Do not import the helper module through the package, with or without a relative
import:

```python
from onnx_light.onnx import helper
from ..onnx import helper
```

If the same file imports other names from `onnx_light.onnx`, keep those names in
a separate `from onnx_light.onnx import ...` statement.

Before completing the change, search all Python sources for imports of `helper`
from `onnx_light.onnx`, including combined and relative imports. Convert every
occurrence and rename its qualified references to `oh.`.
