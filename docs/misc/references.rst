.. _l-references:

References
==========

Papers
++++++

- *SONNX: Towards an ONNX Profile for critical systems*,
  presented by the SONNX Working Group hosted by the ONNX community.
  The paper introduces a restricted, well-defined subset (a *profile*)
  of ONNX whose operators, data types and constructs have clear,
  deterministic and analyzable behaviour, so that ONNX models can be
  used in safety-critical and regulated domains (automotive, medical,
  avionics, industrial control) and aligned with standards such as
  ISO 26262, IEC 62304 or DO-178C.
  Available at `hal-05513194 <https://hal.science/hal-05513194>`_.

  Supporting this profile in ``onnx-light`` would typically involve:

  - declaring which ONNX operators, opsets and data types are part of
    the SONNX profile (for example through additional metadata on
    ``LightOpSchema`` or a dedicated allow-list);
  - providing a checker pass that rejects models using operators,
    attributes or types outside of the profile;
  - documenting, for every reference kernel in
    ``onnx_backend_test``, whether its numerical behaviour is fully
    deterministic and bit-reproducible across platforms;
  - exposing the corresponding artifacts (conformance test reports,
    traceability matrices) needed for certification.

  See also the `ONNX roadmap
  <https://github.com/onnx/onnx/blob/main/ROADMAP.md>`_ for the upstream
  project's direction and priorities, which provides useful context for
  how a SONNX profile could evolve alongside ONNX.
