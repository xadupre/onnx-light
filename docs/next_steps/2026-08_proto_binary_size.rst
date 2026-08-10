.. _l-next-steps-proto-binary-size:

Reducing the ``lib_onnx_proto`` binary size
===========================================

:Date: 2026-08

**complete**

Objective
+++++++++

The CMake target ``lib_onnx_proto`` produces ``liblib_onnx_proto.so`` (the shared
library linked by consumers that only need to parse and serialize ONNX models).
That dependency should not pull in unrelated features or pay for one copy of
every convenience wrapper generated for every message class.

The primary objective is to reduce both:

* the installed size of ``onnx_light/onnx_py/liblib_onnx_proto.so``;
* the mapped code and read-only data needed at runtime.

Size reduction must preserve the ONNX wire format, parser and serializer
behavior, external-data support required by ordinary models, and the ability
to exchange proto objects across onnx-light shared libraries.

Measured baseline
+++++++++++++++++

The current Linux x86-64 Release library was measured with OpenSSL enabled:

.. list-table::
   :header-rows: 1
   :widths: 42 24 34

   * - Metric
     - Size
     - Observation
   * - File on disk
     - 2,178,120 bytes
     - The ELF file is not stripped
   * - File after ``strip --strip-unneeded``
     - 1,810,208 bytes
     - Immediate 16.9% packaging reduction
   * - Allocated ELF sections
     - 1,803,125 bytes
     - Approximate mapped image
   * - ``.text``
     - 1,161,276 bytes
     - 64.4% of allocated sections
   * - ``.dynstr`` + ``.dynsym`` + ``.gnu.hash``
     - 271,559 bytes
     - Cost of the large exported ABI
   * - Unwind and exception tables
     - 211,113 bytes
     - ``.eh_frame*`` and ``.gcc_except_table``
   * - ``.rodata``
     - 40,008 bytes
     - Not the main contributor

The shared object defines approximately 2,119 dynamic symbols. Symbol-name
classification finds about 270 parsing, 629 serialization, 179 size, and 30
printing exports. These groups overlap, but they show that generated
per-message forwarding methods dominate the public surface.

The baseline can be reproduced with:

.. code-block:: bash

    stat --format='%s' onnx_light/onnx_py/liblib_onnx_proto.so
    size -A -d onnx_light/onnx_py/liblib_onnx_proto.so
    readelf --dyn-syms -W onnx_light/onnx_py/liblib_onnx_proto.so
    nm -S --size-sort --demangle \
        onnx_light/onnx_py/liblib_onnx_proto.so

Why the library is large
++++++++++++++++++++++++

Generated message API
^^^^^^^^^^^^^^^^^^^^^

``SERIALIZATION_METHOD`` currently adds a broad API to every proto class:

* two ``ParseFromString`` overloads;
* zero-copy, stream, file-descriptor, array, and iostream parsing;
* string, array, stream, file-descriptor, and iostream serialization;
* size computation and text printing.

Most wrappers perform the same adaptation around a much smaller wire-format
core. Because the shared library exports them for every message class, the
linker must retain them even with ``-ffunction-sections`` and
``--gc-sections``.

Exported ABI
^^^^^^^^^^^^

The target does not use hidden visibility on ELF platforms. Template
instantiations, compatibility wrappers, helper methods, and message methods
therefore enter ``.dynsym`` and ``.dynstr``. The symbol names are particularly
large because many exported functions contain fully expanded C++ template
types.

Mixed responsibilities
^^^^^^^^^^^^^^^^^^^^^^

``lib_onnx_proto`` currently includes more than wire parsing and
serialization:

* model and tensor verification;
* advanced external-data rewriting and alignment;
* encrypted model I/O and the OpenSSL dependency;
* BLAKE3 hashing;
* thread-pool support;
* general ``onnx_light_helpers`` utilities;
* text-format printing.

These features are useful, but a parser/serializer-only consumer should not
have to load all of them.

Compiler-generated metadata
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Exceptions, many out-of-line functions, and the large dynamic ABI generate
substantial unwind, exception, relocation, and procedure-linkage tables.
Removing source code without reducing the number of retained functions and
exports will therefore leave a significant secondary cost.

Optional feature splitting
++++++++++++++++++++++++++

Feature splitting is a separate workstream, not part of the primary binary
size plan below. It changes target composition and dependency ownership, so
its impact must be measured independently from code-generation and linker
improvements.

The minimal shared library should own only:

* message storage and field access needed across shared-library boundaries;
* binary stream primitives;
* wire parsing and serialization;
* ordinary inline and external tensor payload loading;
* the minimal public API needed by the Python bindings and dependent
  onnx-light libraries.

Optional responsibilities should move behind separate targets:

.. list-table::
   :header-rows: 1
   :widths: 32 34 34

   * - Candidate target
     - Responsibility
     - Current sources to examine
   * - ``lib_onnx_proto``
     - Minimal parser, serializer, messages, and streams
     - ``onnx.cc``, ``stream.cc``, required parts of ``simple_string.cc``
   * - ``lib_onnx_proto_tools``
     - Verification and advanced external-data transformations
     - ``onnx_verify.cc``, ``onnx_helper.cc``
   * - ``lib_onnx_proto_hash``
     - Content hashes
     - BLAKE3 sources and hash-specific message methods
   * - ``lib_onnx_proto_crypto``
     - Encrypted model I/O
     - ``onnx_crypt.cc`` and OpenSSL
   * - ``lib_onnx_proto_text``
     - Human-readable proto printing
     - ``PrintToStringStream`` implementations

An ``onnx_proto_full`` interface target may link all components for existing
high-level consumers. The minimal target must not acquire optional
dependencies transitively.

Generated API reduction
+++++++++++++++++++++++

The highest-priority structural change is to stop emitting every convenience
wrapper for every message.

Each message still needs a small type-specific core:

.. code-block:: text

    ParseFromStream(BinaryStream&, ParseOptions&)
    SerializeToStream(BinaryWriteStream&, SerializeOptions&) const
    SerializeSize(BinaryWriteStream&, SerializeOptions&) const

String, array, iostream, zero-copy, and file-descriptor entry points should be
implemented as inline CRTP/mixin wrappers or shared non-template adapters.
Only wrappers used by a consumer are then instantiated in that consumer.

Text printing should not be part of the mandatory serialization macro. It can
be supplied by ``lib_onnx_proto_text`` or enabled explicitly for builds that
need protobuf-compatible debug output.

This refactoring is an ABI change and must be measured separately from
feature splitting. Wire compatibility does not require preserving every
out-of-line convenience symbol.

This step is now implemented with ``ProtoMessageAdapter<T>``. Generated
messages inherit its inline compatibility API, while only
``ParseFromStream``, ``SerializeToStream``, ``SerializeSize``, and text
printing remain substantial type-specific out-of-line methods. ``CopyFrom``
keeps a 30-byte type-specific entry point for ABI compatibility, but delegates
its serialization/deserialization pipeline to one type-erased shared
implementation. The Release build no longer exports one copy of each string,
array, iostream, zero-copy, and file-descriptor adapter for every message.

On the same local Linux Release configuration used for step 2, the defined
dynamic symbol count decreased from 1,191 to 652. The stripped library
decreased from 1,625,024 to 1,001,208 bytes, a reduction of 623,816 bytes
(about 609 KiB).

Visibility and linking
++++++++++++++++++++++

After identifying the cross-library ABI, the ELF and Mach-O builds should use:

.. code-block:: cmake

    set_target_properties(lib_onnx_proto PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN YES)

An ``ONNX_LIGHT_PROTO_API`` annotation or linker version script should expose
only symbols required by public consumers and other onnx-light shared
libraries. Internal templates, helper functions, and implementation details
must remain hidden.

This step is now implemented for ELF and Mach-O builds. The shared target uses
hidden visibility by default, while proto messages, stream types, and the
documented helper API are explicitly marked with ``ONNX_LIGHT_PROTO_API``.
Windows retains ``WINDOWS_EXPORT_ALL_SYMBOLS`` until explicit DLL import/export
annotations are introduced there.

On the local Linux Release baseline, the change reduced the defined dynamic
symbol count from 2,128 to 1,191. The stripped library decreased from
1,810,208 to 1,625,024 bytes, a reduction of 185,184 bytes (about 181 KiB).

The existing function/data sections and dead-section elimination should be
retained. Once visibility is reduced, ``--gc-sections`` can discard code that
is currently kept alive only because it is exported. Identical code folding
may be enabled when supported by the selected linker.

Build and packaging improvements
++++++++++++++++++++++++++++++++

Release wheels should strip unneeded static symbols. This is an immediate
reduction of approximately 368 KB in the measured build and does not change
the runtime ABI.

The following build variants may optionally be compared if additional size
headroom becomes necessary:

* ``Release`` versus ``MinSizeRel``;
* ``-O2`` versus ``-Os``;
* link-time optimization or ThinLTO;
* ``-fno-semantic-interposition`` for hidden internal functions;
* linker identical-code folding.

These options are secondary to API consolidation and feature splitting.
Compiler flags alone cannot remove thousands of intentionally exported
functions.

Static linking remains useful for a standalone parser because the final
linker can retain only referenced sections. Python extensions still need one
shared proto implementation so that message objects and RTTI are not
duplicated across modules.

Measurement plan
++++++++++++++++

Every experiment should record:

* unstripped and stripped file sizes;
* allocated section sizes;
* ``.text``, dynamic symbol/string, relocation, and unwind sizes;
* number of defined dynamic symbols;
* parse and serialization time on the same representative models;
* peak memory while parsing;
* required shared-library dependencies.

Measurements must use a clean build with a recorded compiler, linker, build
type, architecture, ``ONNX_ML`` setting, and OpenSSL setting. Comparing files
from different configurations is not actionable.

Proposed budgets
++++++++++++++++

For the same Linux x86-64 configuration as the baseline:

.. list-table::
   :header-rows: 1
   :widths: 34 28 38

   * - Milestone
     - Installed size
     - Required change
   * - Packaging baseline
     - At most 1.75 MiB
     - Strip release artifacts
   * - Minimal parser/serializer
     - At most 1.25 MiB
     - Reduce exports and generated wrappers
   * - Stretch target
     - At most 1.00 MiB
     - Consolidate generated wrappers and enable size-oriented linking

The parser and serializer must remain wire-compatible. Performance regressions
must be reported alongside size gains rather than hidden by the aggregate
binary-size number.

Implementation order
++++++++++++++++++++

1. **Implemented:** strip the installed Release artifact and report its file,
   section, text, dynamic-symbol, and dependency sizes in CI. See
   `PR #4333 <https://github.com/xadupre/onnx-light/pull/4333>`_.
2. **Implemented:** introduce hidden visibility and an explicit
   ``ONNX_LIGHT_PROTO_API`` cross-library export boundary. See
   `PR #4344 <https://github.com/xadupre/onnx-light/pull/4344>`_.
3. **Implemented:** replace per-message convenience implementations with the
   inline ``ProtoMessageAdapter<T>`` CRTP adapter. See
   `PR #4349 <https://github.com/xadupre/onnx-light/pull/4349>`_.
4. **Optional:** compare ``MinSizeRel``, LTO, and linker folding if additional
   size headroom becomes necessary.
5. **Implemented:** enforce a 1.2 MiB installed-size budget in CI for the Linux
   x86-64 Release build. See
   `PR #4355 <https://github.com/xadupre/onnx-light/pull/4355>`_.

Expected gain by step
+++++++++++++++++++++

The following estimates use the 2.08 MiB unstripped Linux baseline. They are
not additive: visibility, wrapper consolidation, garbage collection, and LTO
may eliminate some of the same code.

.. list-table::
   :header-rows: 1
   :widths: 8 28 20 25 19

   * - Step
     - Change
     - Direct gain
     - Expected resulting size
     - Confidence
   * - 1
     - Strip Release artifacts
     - 359 KiB measured
     - 1.73 MiB
     - High
   * - 2
     - Hide internal symbols and reduce exports
     - 181 KiB measured
     - 1.55 MiB
     - High
   * - 3
     - Share or inline per-message convenience wrappers
     - 609 KiB measured
     - 0.95 MiB
     - High
   * - 4
     - Optional: ``MinSizeRel``, LTO, and identical-code folding
     - 50--150 KiB
     - 0.85--1.15 MiB
     - Low until benchmarked
   * - 5
     - Enforce the CI budget
     - No immediate reduction
     - Prevents regressions
     - High

The optional feature split could save a further 100--300 KiB and remove
dependencies such as OpenSSL from parser-only deployments. It is deliberately
excluded from the cumulative figures because it changes library composition
rather than optimizing the same target.

The measured stripped library is now approximately 0.95 MiB after wrapper
consolidation, meeting the stretch target without the optional compiler and
linker experiments in step 4. CI enforces the 1.2 MiB installed-size budget on
the matching Linux x86-64 Release build.
