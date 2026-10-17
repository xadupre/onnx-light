## Code Standards

### Required Before Each Commit
- Run `black . && ruff check .` before committing any changes to ensure proper Python code formatting
- Run `clang-format -i` on any modified C++ files (`.cpp`, `.h`, `.hpp`, `*.cc`) to ensure proper C++ code formatting.
  The repository `.clang-format` config (LLVM-based, 100-column limit) governs the style.
  To format all C++ files at once: `find . -name "*.cpp" -o -name "*.h" -o -name "*.hpp" -o -name "*.cc" | grep -v ".git" | xargs clang-format -i`
  To check formatting the same way CI does (fails on any diff, matching `.github/workflows/clang_format.yml`):
  `find . \( -name "*.cc" -o -name "*.h" -o -name "*.hpp" \) | grep -v ".git" | xargs clang-format --dry-run --Werror`

### Lightweight Review Before Submission
- Before committing or pushing code changes in this repository, perform a lightweight review of the final diff in addition to the required tests and formatters.
- Keep the review focused on the requested behavior, changed code, direct callers, and directly related tests. Check for regressions, boundary cases, and producer/consumer contract mismatches.
- Include a targeted security check of the changed behavior: untrusted inputs, trust boundaries, path containment and links, bounds and integer overflows, resource limits, and validation bypasses. Check that fast paths and default or placeholder values do not skip required validation. This is not a full security audit.
- Fix confirmed issues caused by the changes and validate the affected behavior before submission. Review any resulting semantic changes; do not repeat an unchanged review before pushing an already-reviewed commit.
- Do not automatically invoke `review-team`, a multi-agent workflow, or a whole-repository audit. The full review team remains available only when explicitly requested.
- Keep unrelated findings out of the change and follow the PR scope rules.

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

### GraphBuilder consistency and performance
- Keep `GraphBuilder` fully coherent after every mutation; callers must immediately observe correct shapes, tags, lifetimes, reuse decisions, and memory metadata.
- Update only the nodes and values affected by a mutation. Do not copy the complete `ShapesContext`, rebuild the whole graph, or defer consistency until export or query time.
- `RefreshAfterMutation` must propagate targeted incremental updates and must not call `BuildGraphImpl` or whole-graph analysis.
- Treat a shape-inference failure during mutation as a programming error; do not copy or roll back `ShapesContext` merely to make the builder reusable after that failure.
- Graph inputs are never overwritable or shadowable. Incremental `ComputeContext` updates must keep them alive rather than offering their storage for in-place reuse.
- Propagate nested-builder mutations from the referencing node through only its affected consumers; never rebuild every node in an ancestor.
