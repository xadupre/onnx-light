.. _l-design-calibration-profile-store:

Backend calibration profile storage
===================================

``CalibrationProfileStore`` centralizes persistent calibration for accelerated
backends. Backend kernel libraries do not perform filesystem access: they build
an exact or portable key, provide policy serialization and validation callbacks,
and call the store.

Identity and selection
++++++++++++++++++++++

Exact keys contain the backend, operator, implementation version, model digest,
processor description (including any cache-topology identity required by the
backend), and effective thread count. Portable keys replace the machine and
model identity with sorted structural model properties.

Lookup precedence is:

1. an exact user override;
2. a portable user override;
3. an exact calibrated profile;
4. a portable calibrated profile.

``force_portable`` skips both exact entries. A lookup returns all rejected
candidates with reasons such as an outdated implementation, a different model
or processor, or a backend validation failure. The selected profile includes
the original named measurements for inspection.

Backend integration
+++++++++++++++++++

The policy is an opaque string. A backend supplies the serializer used by
``Store`` or ``InstallOverride`` and may supply a validation/deserialization
callback to storage and lookup:

.. code-block:: cpp

    CalibrationProfileStore store;
    CalibrationProfileKey key;
    key.backend = "example_gpu";
    key.operator_name = "MatMul";
    key.implementation_version = "3";
    key.model_digest = model_digest;
    key.processor = processor_and_cache_identity;
    key.thread_count = effective_threads;

    store.Store(
        key, {{"median_latency", median_us, "us"}},
        [&policy] { return policy.Serialize(); },
        [](std::string_view payload, std::string &error) {
          return ExamplePolicy::ValidateSerialized(payload, error);
        });

    CalibrationProfileLookupOptions lookup;
    lookup.exact_key = key;
    lookup.structural_properties = {{"dtype", "float16"}, {"shape", "m,n,4096"}};
    auto selected = store.Lookup(lookup, ExamplePolicy::ValidateSerialized);

This is the complete persistence integration: the example backend owns policy
semantics while ``onnx-light`` owns loading, locking, merging, and replacement.
The buildable ``examples/calibration_profile_backend`` program contains the
same integration without any backend-local persistence code.

Persistence and format
++++++++++++++++++++++

The configurable default path is below ``LOCALAPPDATA`` on Windows,
``XDG_CACHE_HOME`` when set, or ``$HOME/.cache`` on other platforms.
``persistence_enabled=false`` disables disk reads and writes while retaining
all in-memory calibration and override behavior.

The version-one text format begins with:

.. code-block:: text

    onnx_light_calibration_profiles 1
    profile exact calibrated
    backend "example_gpu"
    operator "MatMul"
    implementation "3"
    model_digest "..."
    processor "..."
    thread_count 8
    measurement "median_latency" 2.5 "us"
    policy "opaque backend data"
    end

Portable records use ``profile portable`` and repeat ``structure "name"
"value"`` fields instead of exact fields. Overrides use ``override`` in place
of ``calibrated``. Quoted values follow ``std::quoted`` escaping.

Writers take an inter-process lock, reload and merge the latest complete file,
write and synchronize a temporary sibling, then atomically replace the old
file. An interrupted write can therefore leave only an ignored ``.tmp`` file.
Malformed data and unknown schema versions are reported explicitly and never
replace the current in-memory snapshot.
