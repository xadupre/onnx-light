.. _l-next-steps-split-wheels:

Splitting ``onnx-light`` into composable wheels
===============================================

:Date: 2026-08

**discussion**

Objective
+++++++++

The project should publish one wheel per functional library so users install
only the features they need.  For example:

.. code-block:: bash

    pip install onnx-light-proto
    pip install onnx-light-core
    pip install onnx-light-lib

``onnx-light-proto`` provides model reading and writing,
``onnx-light-core`` provides all of ``onnx_light.onnx_core`` and nothing from
the other functional packages, and ``onnx-light-lib`` provides
``onnx_light.onnx_lib``.  The same rule applies to the remaining libraries.

The ``onnx-light`` distribution remains the full installation experience, but
becomes a meta-distribution.  It depends on every public component at exactly
the same version instead of embedding their files again.  Therefore:

.. code-block:: text

    install(onnx-light)
        == install(all public onnx-light-* component wheels)

This equivalence concerns installed features and files.  It must hold on every
supported platform and Python version.

Non-negotiable packaging rules
++++++++++++++++++++++++++++++

Every installed path has exactly one owning wheel.  In particular, two wheels
must never contain the same:

* Python source file;
* Python extension module (``.so``, ``.pyd`` or ``.dylib``);
* shared library (ELF, Mach-O or DLL);
* header, CMake file or package metadata file.

A component wheel may depend on another component and dynamically link its
shared library, but it must not copy that library.  Statically linking a
dependency into several Python extensions would also duplicate compiled code
and is not an acceptable substitute.

All component versions are released in lockstep.  Dependencies between
components use an exact version, for example
``onnx-light-core==0.1.18``.  This prevents an ABI built for one release from
being loaded with shared libraries from another release.

Distribution names use hyphens while import packages keep underscores:
``onnx-light-core`` owns ``onnx_light.onnx_core``.  Component wheels contribute
distinct children to the PEP 420 ``onnx_light`` namespace; they do not each
ship an ``onnx_light/__init__.py`` file.

Proposed wheel matrix
+++++++++++++++++++++

The initial split follows the existing CMake targets and source directories.
The ``Owns`` column is exclusive: no other wheel may package those files.

.. list-table::
   :header-rows: 1
   :widths: 22 30 25 23

   * - Distribution
     - Owns
     - Direct component dependencies
     - Purpose
   * - ``onnx-light-proto``
     - ``onnx_light/onnx_proto``, the proto-only binding, and
       ``lib_onnx_proto``
     - none
     - ONNX message types and model reading and writing, including external
       data; no checker, schema, shape, graph, or runtime functionality
   * - ``onnx-light-core``
     - all of ``onnx_light/onnx_core``, the core binding, and
       ``lib_onnx_core``
     - ``onnx-light-proto``
     - Generic registries, symbolic expressions, graph mechanisms, compute and
       runtime interfaces; no concrete operator implementation
   * - ``onnx-light-manipulations``
     - ``onnx_light/onnx_manipulations`` and
       ``lib_onnx_manipulations``
     - ``onnx-light-core``
     - Parser, printer, compose, and schema-independent graph and tensor
       helpers
   * - ``onnx-light-lib``
     - ``onnx_light/onnx_lib``, its binding, and ``lib_onnx_lib``
     - ``onnx-light-manipulations``
     - Full ONNX schemas and history, checker, inliner, shape inference, and
       version converter
   * - ``onnx-light-op``
     - ``onnx_light/onnx_op``, its binding, and ``lib_onnx_op``
     - ``onnx-light-core``
     - Lightweight ONNX operator schema registrations
   * - ``onnx-light-shape``
     - ``onnx_light/onnx_extensions/shapes``, its binding, and
       ``lib_onnx_shape``
     - ``onnx-light-core``
     - Concrete shape-inference functions and graph optimization helpers
   * - ``onnx-light-patterns``
     - ``onnx_light/onnx_extensions/patterns`` and
       ``lib_onnx_patterns``
     - ``onnx-light-core``
     - Concrete graph-rewriting patterns
   * - ``onnx-light-kernels``
     - ``onnx_light/onnx_extensions/kernels``, its binding, and
       ``lib_onnx_kernels``
     - ``onnx-light-core``
     - Reference runtime and concrete operator kernels
   * - ``onnx-light-backend-test``
     - ``onnx_light/onnx_extensions/backend_test``, its binding, and
       ``lib_onnx_backend_test``
     - ``onnx-light-kernels``
     - Backend test infrastructure and case registries
   * - ``onnx-light-gradient``
     - ``onnx_light/onnx_extensions/gradient``, its binding, and
       ``lib_onnx_gradient``
     - ``onnx-light-core``
     - Reverse-mode graph differentiation
   * - ``onnx-light-tools``
     - ``onnx_light/tools``
     - Only the components imported by each tool
     - Visualization, pretty-printing, schema comparison, and command-line
       tools
   * - ``onnx-light-compat``
     - ``onnx_light/onnx``, ``onnx_light/_reference`` and compatibility
       helpers
     - ``proto``, ``core``, ``manipulations``, ``lib``, ``op`` and ``shape``
     - Drop-in Python compatibility surface for the upstream ``onnx`` API
   * - ``onnx-light``
     - Distribution metadata only
     - Every public component above
     - Full installation and backward-compatible user entry point

The exact dependency graph follows the shared-library links:

.. code-block:: text

    proto
      |
      +-- core
      |     +-- manipulations -- lib
      |     +-- op
      |     +-- shape
      |     +-- patterns
      |     +-- kernels -- backend-test
      |     +-- gradient
      |
      +-- full Python compatibility and tools through their declared needs

Transitive dependencies are not repeated in wheel metadata.  For example,
``onnx-light-lib`` depends directly on ``onnx-light-manipulations``; that
component already brings ``core`` and ``proto``.

Required binding refactoring
++++++++++++++++++++++++++++

The current Python extensions do not yet respect these ownership boundaries.
``_onnxpyprotoop`` combines proto classes, operator schemas, and manipulation
helpers.  ``_onnxpycore`` links ``lib_onnx_shape`` rather than only
``lib_onnx_core``.  Those combinations would force one wheel either to include
another component or to duplicate an extension.

Before producing component wheels, bindings must be split by the same
boundaries as the CMake libraries:

.. code-block:: text

    _onnxpyproto          -> lib_onnx_proto
    _onnxpycore           -> lib_onnx_core
    _onnxpymanipulations  -> lib_onnx_manipulations
    _onnxpylib            -> lib_onnx_lib
    _onnxpyop             -> lib_onnx_op
    _onnxpyshape          -> lib_onnx_shape
    _onnxpypatterns       -> lib_onnx_patterns
    _onnxpykernels        -> lib_onnx_kernels
    _onnxpybackend        -> lib_onnx_backend_test
    _onnxpygradient       -> lib_onnx_gradient

Each extension is installed by the wheel that owns its linked library.
Cross-component Python APIs import the owning extension instead of registering
the same C++ type a second time.  Proto classes in particular have one
nanobind registration, owned by ``onnx-light-proto``.

The native ``lib_onnx_proto`` target must also be reduced to the same
read/write contract.  Message storage, construction helpers, binary and text
serialization, external-data I/O, NumPy conversion, and encrypted I/O remain
in ``proto``.  Validation such as ``onnx_light.onnx_proto.verify`` moves to
``onnx-light-lib`` because it interprets ONNX model semantics.  Schema lookup,
checking, shape inference, graph transformations, patterns, kernels, and
backend tests are likewise excluded.  Compatibility re-exports may preserve
old import paths only in ``onnx-light-compat``; they must not make
``onnx-light-proto`` depend on those features.

``onnx_light.onnx_core`` currently contains a few feature-level Python
facades, such as gradient helpers.  To satisfy the rule that
``onnx-light-core`` contains all and only ``onnx_core``, these facades must
either remain dependency-free core interfaces or move to the component they
implement.  No file under ``onnx_light/onnx_core`` may be conditionally copied
by a second wheel.

Build and installation design
+++++++++++++++++++++++++++++

The build needs an explicit component selector instead of a growing set of
feature booleans:

.. code-block:: text

    ONNX_LIGHT_COMPONENT=proto|core|manipulations|lib|op|shape|patterns|
                         kernels|backend-test|gradient

Selecting a component builds its target and the targets needed for linking,
but installs only the selected target's shared library and binding.  Runtime
dependencies come from their own wheels.  CMake install components should
encode this distinction so a dependency built in the same build tree cannot
leak into the produced wheel.

Python packaging should use one small project definition per distribution,
generated from a single component manifest.  That manifest is the source of
truth for:

* distribution name and version;
* owned Python packages;
* CMake target and extension module;
* direct wheel dependencies;
* optional console entry points.

The build creates every wheel from a clean install staging directory.  Reusing
one populated staging directory risks silently carrying a library from the
previous component.

Release automation on tag creation
++++++++++++++++++++++++++++++++++

The split must preserve the existing automated release contract.  Today,
``.github/workflows/build_release_wheel.yml`` is triggered by every pushed
tag, builds the platform wheels with ``cibuildwheel``, and attaches them to the
corresponding GitHub Release.  Creating a release tag must continue to create
all wheels automatically; no component is built or uploaded manually.

The workflow becomes a two-dimensional build matrix:

.. code-block:: text

    component x platform/architecture/Python ABI

The component dimension comes from the machine-readable component manifest,
not from a second hard-coded list in YAML.  Every matrix job checks out the
tag, selects one ``ONNX_LIGHT_COMPONENT``, and uploads its wheel as a workflow
artifact.  The pure metadata ``onnx-light`` wheel is built once because it is
platform-independent.

Uploading directly to the GitHub Release from every matrix job is unsafe for
the split: a failed component could leave a release containing only part of a
mutually versioned set.  A final release job must:

1. depend on every component and platform build;
2. download all workflow artifacts;
3. verify that every expected wheel exists exactly once;
4. verify tag, project, and dependency versions;
5. run the no-common-assembly and aggregate-equivalence checks;
6. create or locate the GitHub Release and upload the complete wheel set.

No release asset is uploaded before these gates pass.  If publication to PyPI
is added or already performed outside this repository, the same final job
publishes the complete validated directory in one trusted-publishing step.
The aggregate ``onnx-light`` wheel is included only after every exact-version
dependency it declares is present.

Tag validation is mandatory.  A tag such as ``v0.1.18`` must match the shared
component version ``0.1.18`` before expensive matrix jobs start.  Component
wheels and the aggregate wheel are immutable release artifacts; rerunning a
tag workflow may restore missing GitHub Release assets but must not silently
replace an artifact already published to a package index.

The existing workflows should evolve as follows:

.. list-table::
   :header-rows: 1
   :widths: 28 32 40

   * - Workflow
     - Existing role
     - Role after the split
   * - ``build_release_wheel.yml``
     - Builds the monolithic wheel on a tag push
     - Builds every component matrix and the aggregate, validates the complete
       set, then uploads it from one final job
   * - ``build_weekly_wheels.yml``
     - Detects platform and toolchain regressions
     - Builds the same component matrix without creating a release, including
       ownership and installation tests
   * - ``build_reduced_wheel.yml``
     - Builds a second reduced variant on tags and weekly
     - Is removed once equivalent component sets exist, or temporarily becomes
       an explicitly documented aggregate profile without duplicated binaries
   * - ``build_release_cpp.yml``
     - Publishes the complete C++ archive when a GitHub Release is published
     - Remains independent unless component C++ archives are also desired

Platform loading
^^^^^^^^^^^^^^^^

Component libraries cannot all be copied beside every extension.  They need a
single private runtime location owned by their respective wheels, for example
``onnx_light/.libs`` with globally unique library names that include the
project ABI major.  The Linux ``RPATH``, macOS install names, and Windows DLL
search setup must resolve dependencies from that shared location.

Repair tools such as ``auditwheel`` or ``delocate`` must be configured not to
vendor another ``onnx-light`` component library into the wheel being repaired.
The final repaired wheel, not only the pre-repair staging tree, is checked for
duplicate ownership.

Execution plan
++++++++++++++

Phase 1: freeze ownership and ABI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Create the machine-readable component manifest and derive the table above
   from the actual CMake target graph.
2. Assign every installed Python file, extension, and shared library to one
   component; fail if a path has zero or multiple owners.
3. Move verification and every other non-I/O responsibility out of
   ``lib_onnx_proto`` and ``onnx_light.onnx_proto``.
4. Define the cross-library ABI, symbol visibility, library naming, runtime
   directory, and exact-version policy.

Phase 2: separate Python bindings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Split ``_onnxpyprotoop`` into proto, manipulation, and operator bindings.
2. Make the core binding link only ``lib_onnx_core`` and add a separate shape
   binding.
3. Add bindings for components that expose Python APIs but currently share an
   extension.
4. Move misplaced Python facades to their owning component while preserving
   compatibility imports in ``onnx-light-compat``.
5. Test every component in isolation before changing release packaging.

Phase 3: component-aware CMake install
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Add one CMake install component per public library and binding.
2. Separate ``build dependency`` from ``install artifact`` so only the selected
   component enters its staging directory.
3. Set cross-platform runtime lookup paths and verify that extensions consume,
   rather than vendor, dependency libraries.
4. Keep the existing all-target developer build for source-tree development.

Phase 4: build component wheels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Add one packaging project per component, all reading the shared version and
   manifest.
2. Extend the weekly workflow with the component matrix before changing the
   tag-triggered release workflow.
3. Build wheels in dependency order from clean directories.
4. Install each wheel alone into a clean environment and run its focused tests.
5. Install dependency chains such as ``proto -> core -> shape`` and
   ``proto -> core -> kernels -> backend-test``.
6. Inspect wheel contents after platform repair and compare hashes for every
   installed path.

Phase 5: make ``onnx-light`` the aggregate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Replace the monolithic ``onnx-light`` wheel contents with exact dependencies
   on all public components.
2. Move the root package initialization and compatibility imports out of the
   shared namespace path so the meta-wheel owns no runtime file.
3. Verify that installing ``onnx-light`` and installing the complete component
   set produce identical file manifests and import behavior.
4. Preserve the current command-line and ``import onnx_light`` behavior through
   the compatibility and tools distributions.
5. Replace the tag workflow's per-platform release uploads with one final,
   gated upload of all component and aggregate wheels.

Phase 6: release and migration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Publish all component wheels for one release candidate version before the
   aggregate wheel.
2. Test installation from the package index on Linux, macOS, and Windows,
   including an environment upgraded from the previous monolithic release.
3. Document that old monolithic files must be removed during the transition;
   an upgrade must not leave unowned shared libraries behind.
4. Push a release-candidate tag and prove that the workflow creates the full
   expected artifact set without manual intervention.
5. Publish all artifacts as one atomic release and prohibit partial
   component-version updates.

Acceptance criteria
+++++++++++++++++++

The split is complete when CI proves all of the following:

* ``onnx-light-proto`` can read and write models without installing schemas,
  shapes, kernels, or backend tests;
* ``onnx-light-core`` installs all of ``onnx_light.onnx_core`` and no package
  owned by another component;
* every specialized import fails with a clear missing-component message when
  its wheel is absent;
* every wheel contains exactly one component's compiled artifacts;
* the intersection of installed file paths from any two wheels is empty,
  excluding standard packaging metadata in separate ``.dist-info``
  directories;
* no repaired wheel vendors an ``onnx-light`` shared library owned by another
  wheel;
* all loaded ``onnx-light`` shared libraries have one filesystem instance and
  one in-process image;
* ``pip check`` succeeds for individual dependency chains and the aggregate;
* the union of all component manifests is byte-for-byte identical to the
  runtime manifest installed by ``onnx-light``;
* the existing full test suite passes when only the aggregate
  ``onnx-light`` distribution is requested;
* pushing one valid release tag automatically builds and uploads the complete
  expected wheel matrix, while any failed or missing component prevents every
  release upload.

The manifest-intersection and aggregate-equivalence checks are release gates.
They prevent a future binding or platform repair change from silently
reintroducing a shared assembly into multiple wheels.
