## Code Standards

### Required Before Each Commit
- Run `black . && ruff check .` before committing any changes to ensure proper code formatting

### Docstring Style
- Write docstrings using **third-person singular** verbs (e.g., "Returns the value.", "Computes the output shape.", not "Return the value." or "Compute the output shape.")
- Use **`Returns:`** as the section header for return value documentation, never `Return:`
- One-liner docstrings must end with a period

### Imports
- Avoid using aliases when importing as much as possible

### Private members
- Do not make everything private.

### Exception
- Do not use `try/except` for normal control flow.
- Prefer precondition checks when feasible, but use `try/except` for genuinely exceptional conditions and around external I/O boundaries where failures can still occur.
- Catch specific exceptions rather than using broad exception handlers, and add tests that cover expected failure paths.
