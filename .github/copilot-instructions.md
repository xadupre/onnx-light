# Code Style

This repository enforces code style with [Black](https://black.readthedocs.io/) and
[Ruff](https://docs.astral.sh/ruff/). Style checks run automatically on every pull request
via `.github/workflows/style.yml`.

## Formatter: Black

- Line length: **88 characters**
- Target versions: Python 3.10 through 3.13

Run locally:

```bash
black .
```

Check without modifying files:

```bash
black --check .
```

## Linter: Ruff

- Line length: **88 characters**
- Enabled rule sets: `E`, `F`, `W`, `I` (pycodestyle errors/warnings, pyflakes, isort)
- Ignored rules: `E501` (line too long), `I001` (import block unsorted)

Run locally:

```bash
ruff check .
```

Auto-fix where possible:

```bash
ruff check --fix .
```

## File Header

Every Python source file must begin with the standard Microsoft copyright header:

```python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
```

## Imports

- Standard library imports come first, then third-party, then local imports.
- Each group is separated by a blank line.
- Within a group, imports are sorted alphabetically.

Example:

```python
import os
import unittest

import numpy as np

from .base import Model
```

## Naming

- Classes: `UpperCamelCase` (e.g., `OnnxModel`, `GraphProto`)
- Functions and methods: `snake_case` (e.g., `load_model`, `to_proto`)
- Constants and configuration keys: `snake_case` strings or `UPPER_SNAKE_CASE` for
  module-level constants

## Tests

- Tests live under `tests/fast/` (CI-safe, no network).
- Test classes extend `unittest.TestCase`.
- Use `unittest.main()` at the bottom of each test file.

Run fast tests:

```bash
pytest tests/fast -v
```
