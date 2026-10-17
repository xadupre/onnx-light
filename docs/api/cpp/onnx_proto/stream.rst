.. _l-api-onnx-onnx-proto-stream:

stream.h
========

External-data opening follows the ONNX descriptor-based approach: Linux
``openat2`` with ``RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS`` when available,
``O_RESOLVE_BENEATH`` on supporting platforms, and pre-/post-open containment
and file-identity verification otherwise. Open descriptors are retained for
subsequent reads, writes and mappings; their ownership uses standard
``std::shared_ptr<int>`` rather than a separate file class.
These helpers are implementation details: their header is neither installed
nor included in the public API reference.

Windows and the POSIX fallback do not prevent every directory-replacement race
during acquisition: creation can have an external effect before the post-open
check rejects the operation. Retaining a verified descriptor prevents later
pathname replacements from redirecting I/O, but does not remove that acquisition
limitation, tracked in :issue:`5304`.

.. doxygenfile:: stream.h
   :project: onnx-light
